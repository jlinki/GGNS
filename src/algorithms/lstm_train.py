import copy
import torch
import os
from torch_geometric.loader import DataLoader
import torch.nn as nn
from tqdm import tqdm
import wandb
from util.Types import *
import numpy as np

from src.utils.eval_utils import mpc_m_mesh_evaluation, get_radius_dict_for_evaluation_mode, calculate_mesh_iou
from src.utils.train_utils import get_network_config, log_gradients, seed_worker, seed_all, \
    add_noise_to_mesh_nodes, add_pointcloud_dropout, calculate_loss_normalizer
from src.utils.data_utils import get_timestep_data, convert_to_hetero_data, get_feature_info_from_data, get_shortest_trajectory, \
    count_parameters, convert_trajectory_to_data_list, crop_list_of_trajectories, predict_velocity, transform_position_to_edges
from src.utils.wandb_utils import wandb_init, wandb_loss_logger, wandb_config_update
from src.utils.get_connectivity_setting import get_connectivity_setting
from modules.datasets.Datasets import WeightedSequenceNoReturnDataset, SequenceNoReturnDataset
from modules.gnn_modules.homogeneous_modules.GNNBase import GNNBase
from modules.gnn_modules.heterogeneous_modules.HeteroGNNBase import HeteroGNNBase


from src.utils.dataset_utils import build_dataset_for_split, get_mesh_triangles_from_sofa
from src.utils.graph_utils import create_graph_from_raw


class LstmHomoGNN(nn.Module):
    def __init__(self,
                 in_node_features: Dict[str, int],
                 in_edge_features: Dict[str, int],
                 in_global_features: int,
                 out_node_features: int,
                 network_config: ConfigDict):
        super().__init__()

        self.latent_dimension = network_config.get("latent_dimension")
        self.mlp_decoder = network_config.get("mlp_decoder")

        self._gnn_base = GNNBase(in_node_features=in_node_features,
                                 in_edge_features=in_edge_features,
                                 in_global_features=in_global_features,
                                 network_config=network_config)
        self.lstm_layer = nn.LSTM(self.latent_dimension, self.latent_dimension, 1,
                                  batch_first=True)  # input, hidden, num_layers, batchfirst -> N, L, H_in
        # define decoder
        if self.mlp_decoder:
            self.node_decoder1 = nn.Linear(self.latent_dimension, self.latent_dimension)
            self.node_decoder2 = nn.Linear(self.latent_dimension, out_node_features)
            self.activation = nn.LeakyReLU()
        else:
            self.node_decoder = nn.Linear(self.latent_dimension, out_node_features)

    def forward(self, tensor, h_0, c_0):
        node_features, _, _ = self._gnn_base(tensor)  # N*F, H_in
        node_features = predict_velocity(node_features, tensor, False, False)  # N*F_mesh, H_in
        if h_0 is None:
            node_features, (h_n, c_n) = self.lstm_layer(node_features.view(-1, 1, self.latent_dimension))
        else:
            node_features, (h_n, c_n) = self.lstm_layer(node_features.view(-1, 1, self.latent_dimension),
                                                        (h_0, c_0))  # N*F_mesh, 1, H_out
        if self.mlp_decoder:
            node_features = self.node_decoder1(node_features)
            node_features = self.activation(node_features)
            node_features = self.node_decoder2(node_features)
        else:
            node_features = self.node_decoder(node_features).squeeze()  # N*F_mesh, 1, output_dimension

        return node_features, h_n, c_n


class LstmHeteroGNN(nn.Module):
    def __init__(self,
                 in_node_features: Dict[str, int],
                 in_edge_features: Dict[str, int],
                 in_global_features: int,
                 out_node_features: int,
                 network_config: ConfigDict):
        super().__init__()

        self.latent_dimension = network_config.get("latent_dimension")
        self.mlp_decoder = network_config.get("mlp_decoder")

        self._hetero_gnn_base = HeteroGNNBase(in_node_features=in_node_features,
                                 in_edge_features=in_edge_features,
                                 in_global_features=in_global_features,
                                 network_config=network_config)
        self.lstm_layer = nn.LSTM(self.latent_dimension, self.latent_dimension, 1,
                                  batch_first=True)  # input, hidden, num_layers, batchfirst -> N, L, H_in
        # define decoder
        if self.mlp_decoder:
            self.node_decoder1 = nn.Linear(self.latent_dimension, self.latent_dimension)
            self.node_decoder2 = nn.Linear(self.latent_dimension, out_node_features)
            self.activation = nn.LeakyReLU()
        else:
            self.node_decoder = nn.Linear(self.latent_dimension, out_node_features)

    def forward(self, tensor, h_0, c_0):
        node_features_dict, _, _, _ = self._hetero_gnn_base(tensor) # N*F, H_in
        node_features = predict_velocity(node_features_dict['mesh'], tensor, False, False)  # N*F_mesh, H_in
        if h_0 is None:
            node_features, (h_n, c_n) = self.lstm_layer(node_features.view(-1, 1, self.latent_dimension))
        else:
            node_features, (h_n, c_n) = self.lstm_layer(node_features.view(-1, 1, self.latent_dimension),
                                                        (h_0, c_0))  # N*F_mesh, 1, H_out
        if self.mlp_decoder:
            node_features = self.node_decoder1(node_features)
            node_features = self.activation(node_features)
            node_features = self.node_decoder2(node_features)
        else:
            node_features = self.node_decoder(node_features).squeeze()  # N*F_mesh, 1, output_dimension

        return node_features, h_n, c_n


def get_gnn_model(in_node_features, in_edge_features, in_global_features, out_node_features, network_config, hetero,
                  device):
    if hetero:
        GNN = LstmHeteroGNN(in_node_features=in_node_features,
                        in_edge_features=in_edge_features,
                        in_global_features=in_global_features,
                        out_node_features=out_node_features,
                        network_config=network_config).to(device)
    else:
        GNN = LstmHomoGNN(in_node_features=in_node_features,
                      in_edge_features=in_edge_features,
                      in_global_features=in_global_features,
                      out_node_features=out_node_features,
                      network_config=network_config).to(device)
    return GNN


def single_step_evaluation(model,
                           hetero: bool,
                           evalloader,
                           device,
                           epoch: int,
                           loss_normalizer: float,
                           euclidian_distance=False,
                           use_color=False,
                           tissue_task=False,
                           use_world_edges=False,
                           use_mesh_coordinates=False,
                           mgn_hetero=False) -> float:
    """
    Function to evaluate our model on single step predictions. This means the groundtruth mesh is used as input in every time step. Batched data can be used.
    Args:
        model: The torch.nn.Module model to evaluate
        hetero: True if heterogeneous data is used
        evalloader: The Dataloader for the evaluation data
        device: Working device, either cpu or cuda
        epoch: Current epoch
        loss_normalizer: Normalizer for the loss
        euclidian_distance: True if Euclidean norm included as feature
        use_color: Color gradient texture for point cloud
        tissue_task: True if 3D data is used
        use_world_edges: Use of explicit world edges (see MGN)
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        mgn_hetero: Use small hetero GNN like in MGN
    Returns:
        eval_loss: Mean MSE loss over the predictions of all time steps in all trajectories
    """
    model.eval()
    pbar = tqdm(total=len(evalloader))
    pbar.set_description(f'Single Step Eval epoch: {epoch:04d}')
    eval_loss = 0
    with torch.no_grad():
        for data_list in evalloader:
            target = []
            old_position = []
            velocity_list = []
            c_0 = None
            h_0 = None
            for data in data_list:
                data.to(device)
                data = transform_position_to_edges(data, euclidian_distance)
                data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                node_features_out, h_0, c_0 = model(data, h_0, c_0)
                velocity_list.append(node_features_out)
                target.append(data.y)
                old_position.append(data.y_old)
            target = torch.stack(target, dim=1)
            old_position = torch.stack(old_position, dim=1)
            velocity = torch.cat(velocity_list, dim=1)
            predicted_position = old_position + velocity
            criterion = nn.MSELoss()
            loss = criterion(predicted_position, target)
            eval_loss += loss
            pbar.update(1)
    pbar.close()
    eval_loss = eval_loss.item()/len(evalloader)/loss_normalizer
    print("single step eval loss: ", eval_loss)

    return eval_loss


def n_step_evaluation_iou(model,
                      hetero: bool,
                      n: int,
                      trajectory_list_eval_raw: list,
                      device,
                      epoch: int,
                      loss_normalizer: float,
                      edge_radius_dict: Dict,
                      input_timestep="t+1",
                      mode=None,
                      k=0,
                      euclidian_distance=False,
                      use_color=False,
                      use_poisson=False,
                      tissue_task=False,
                      use_world_edges=False,
                      use_mesh_coordinates=False,
                      mgn_hetero=False,
                      mesh_triangles=None) -> tuple:
    """
    Function to evaluate our model on n-step predictions. This means the ground truth mesh is used as input in every n-th time step.
    Args:
        model: The torch.nn.Module model to evaluate
        hetero: True if heterogeneous data is used
        n: Number of time steps after which a ground truth mesh is input again
        trajectory_list_eval_raw: The Dataloader for the evaluation data
        device: Working device, either cpu or cuda
        epoch: Current epoch
        loss_normalizer: Normalizer for the loss
        edge_radius_dict: Dictionary containing the edge radii for the used connectivity setting
        input_timestep: Either 't' or 't+1' if point cloud and collider of future time step is used
        mode: Evaluation mode, default is using the edge_radius_dict, other modes use mesh-only in some steps
        k: Additional argument for mode
        euclidian_distance: True if Euclidean norm included as feature
        use_color: Color gradient texture for point cloud
        use_poisson: True if the poisson's ratio is used as feature
        tissue_task: True if 3D data is used
        use_world_edges: Use of explicit world edges (see MGN)
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        mgn_hetero: Use small hetero GNN like in MGN
        calculate_iou: calculate the mesh IOU between prediction and ground truth
    Returns:
        eval_loss: Mean MSE loss over the predictions of all time steps in all trajectories
    """
    if tissue_task:
        eval_loss = n_step_evaluation(model,
                      hetero,
                      n,
                      trajectory_list_eval_raw,
                      device,
                      epoch,
                      loss_normalizer,
                      edge_radius_dict,
                      input_timestep,
                      mode,
                      k,
                      euclidian_distance,
                      use_color,
                      use_poisson,
                      tissue_task,
                      use_world_edges,
                      use_mesh_coordinates,
                      mgn_hetero)
        eval_iou = 0.0
    else:
        model.eval()
        pbar = tqdm(total=len(trajectory_list_eval_raw))
        pbar.set_description(f'{n}_step eval loss: {epoch:04d}')
        eval_loss = 0
        eval_iou = 0
        with torch.no_grad():
            for data_list in trajectory_list_eval_raw:
                h_0 = None
                c_0 = None
                for index, data in enumerate(data_list):

                    # inputs the ground truth mesh after n timesteps
                    if index % n == 0:
                        predicted_position = data.y_old.to(device)

                    # get correct data for current timestep from raw data and build graph
                    data_timestep = get_timestep_data(data, predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                    current_edge_radius_dict = get_radius_dict_for_evaluation_mode(index, edge_radius_dict, mode, k)
                    data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates, hetero=hetero, tissue_task=tissue_task)
                    data = transform_position_to_edges(data, euclidian_distance)
                    data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                    data.ptr = torch.tensor([0, data.batch.shape[0]])  # needed if mgn_hetero is used
                    data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                    data.to(device)

                    # evaluate model
                    velocity, h_0, c_0 = model(data, h_0, c_0)
                    old_position = predicted_position
                    predicted_position = old_position + velocity.squeeze()
                    criterion = nn.MSELoss()
                    loss = criterion(predicted_position, data.y)
                    mesh_iou = calculate_mesh_iou(data.y.cpu().numpy(), predicted_position.detach().cpu().numpy(), mesh_triangles, mesh_triangles)
                    eval_loss += loss
                    eval_iou += mesh_iou
                pbar.update(1)
        pbar.close()
        eval_loss = eval_loss.item()/len(trajectory_list_eval_raw)/len(data_list)/loss_normalizer
        eval_iou = eval_iou.item()/len(trajectory_list_eval_raw)/len(data_list)/loss_normalizer
        print(str(n) + "_step eval loss: ", eval_loss)

    return eval_loss, eval_iou


def n_step_evaluation(model,
                      hetero: bool,
                      n: int,
                      trajectory_list_eval_raw: list,
                      device,
                      epoch: int,
                      loss_normalizer: float,
                      edge_radius_dict: Dict,
                      input_timestep="t+1",
                      mode=None,
                      k=0,
                      euclidian_distance=False,
                      use_color=False,
                      use_poisson=False,
                      tissue_task=False,
                      use_world_edges=False,
                      use_mesh_coordinates=False,
                      mgn_hetero=False) -> float:
    """
    Function to evaluate our model on n-step predictions. This means the ground truth mesh is used as input in every n-th time step.
    Args:
        model: The torch.nn.Module model to evaluate
        hetero: True if heterogeneous data is used
        n: Number of time steps after which a ground truth mesh is input again
        trajectory_list_eval_raw: The Dataloader for the evaluation data
        device: Working device, either cpu or cuda
        epoch: Current epoch
        loss_normalizer: Normalizer for the loss
        edge_radius_dict: Dictionary containing the edge radii for the used connectivity setting
        input_timestep: Either 't' or 't+1' if point cloud and collider of future time step is used
        mode: Evaluation mode, default is using the edge_radius_dict, other modes use mesh-only in some steps
        k: Additional argument for mode
        euclidian_distance: True if Euclidean norm included as feature
        use_color: Color gradient texture for point cloud
        use_poisson: True if the poisson's ratio is used as feature
        tissue_task: True if 3D data is used
        use_world_edges: Use of explicit world edges (see MGN)
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        mgn_hetero: Use small hetero GNN like in MGN
    Returns:
        eval_loss: Mean MSE loss over the predictions of all time steps in all trajectories
    """
    model.eval()
    pbar = tqdm(total=len(trajectory_list_eval_raw))
    pbar.set_description(f'{n}_step eval loss: {epoch:04d}')
    eval_loss = 0
    with torch.no_grad():
        for data_list in trajectory_list_eval_raw:
            h_0 = None
            c_0 = None
            for index, data in enumerate(data_list):

                # inputs the ground truth mesh after n timesteps
                if index % n == 0:
                    predicted_position = data.y_old.to(device)

                # get correct data for current timestep from raw data and build graph
                data_timestep = get_timestep_data(data, predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                current_edge_radius_dict = get_radius_dict_for_evaluation_mode(index, edge_radius_dict, mode, k)
                data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates, hetero=hetero, tissue_task=tissue_task)
                data = transform_position_to_edges(data, euclidian_distance)
                data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                data.ptr = torch.tensor([0, data.batch.shape[0]])  # needed if mgn_hetero is used
                data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                data.to(device)

                # evaluate model
                velocity, h_0, c_0 = model(data, h_0, c_0)
                old_position = predicted_position
                predicted_position = old_position + velocity.squeeze()
                criterion = nn.MSELoss()
                loss = criterion(predicted_position, data.y)
                eval_loss += loss
            pbar.update(1)
    pbar.close()
    eval_loss = eval_loss.item()/len(trajectory_list_eval_raw)/len(data_list)/loss_normalizer
    print(str(n) + "_step eval loss: ", eval_loss)

    return eval_loss


def mpc_m_mesh_evaluation_iou(model,
                      hetero: bool,
                      path_mesh_length: int,
                      path_stride: int,
                      trajectory_list_eval_raw: list,
                      device,
                      epoch: int,
                      loss_normalizer: float,
                      edge_radius_dict: Dict,
                      input_timestep="t+1",
                      mode_pcd="full-pcd",
                      mode_mesh="mesh-only",
                      euclidian_distance=False,
                      use_color=False,
                      use_poisson=False,
                      tissue_task=False,
                      use_world_edges=False,
                      use_mesh_coordinates=False,
                      mgn_hetero=False,
                      mesh_triangles=None) -> tuple:
    """
    Function to evaluate our model on T-step predictions with T <= len(traj). path_mesh_length steps include no point cloud.
    Args:
        model: The torch.nn.Module model to evaluate
        hetero: True if heterogeneous data is used
        path_mesh_length: Number of time steps without point cloud but mesh only
        path_stride: Stride for T-step predictions
        trajectory_list_eval_raw: The Dataloader for the evaluation data
        device: Working device, either cpu or cuda
        epoch: Current epoch
        loss_normalizer: Normalizer for the loss
        edge_radius_dict: Dictionary containing the edge radii for the used connectivity setting
        input_timestep: Either 't' or 't+1' if point cloud and collider of future time step is used
        mode_pcd: Evaluation mode for time steps with point cloud
        mode_mesh: Evaluation mode for time steps without point cloud
        euclidian_distance: True if Euclidean norm included as feature
        use_color: Color gradient texture for point cloud
        use_poisson: True if the poisson's ratio is used as feature
        tissue_task: True if 3D data is used
        use_world_edges: Use of explicit world edges (see MGN)
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        mgn_hetero: Use small hetero GNN like in MGN
    Returns:
        eval_loss: Mean MSE loss over the predictions of all time steps in all trajectories
    """
    if tissue_task:
        eval_loss = mpc_m_mesh_evaluation(model,
                      hetero,
                      path_mesh_length,
                      path_stride,
                      trajectory_list_eval_raw,
                      device,
                      epoch,
                      loss_normalizer,
                      edge_radius_dict,
                      input_timestep,
                      mode_pcd,
                      mode_mesh,
                      euclidian_distance,
                      use_color,
                      use_poisson,
                      tissue_task,
                      use_world_edges,
                      use_mesh_coordinates,
                      mgn_hetero)
        eval_iou = 0.0
    else:
        model.eval()
        criterion = nn.MSELoss()
        pbar = tqdm(total=len(trajectory_list_eval_raw))
        pbar.set_description(f'MPC eval {path_mesh_length}-mesh loss: {epoch:04d}')
        len_traj = len(trajectory_list_eval_raw[0])
        max_traj_index = len_traj - path_mesh_length
        path_start_indices = np.arange(0, max_traj_index + 1, path_stride)
        num_paths = len(path_start_indices)
        eval_loss = 0
        eval_iou = 0

        with torch.no_grad():
            for traj_i in range(len(trajectory_list_eval_raw)):
                traj_loss = 0
                traj_iou = 0
                path_pcd_loss = 0
                path_pcd_iou = 0
                h_0 = None
                c_0 = None
                h_pcd = h_0
                c_pcd = h_0
                predicted_position = trajectory_list_eval_raw[traj_i][0].y_old
                last_prediction_with_pcd = predicted_position

                for path_number in range(num_paths):
                    path_loss = 0
                    path_iou = 0
                    path_pcd_length_upper = path_start_indices[path_number]

                    if path_pcd_length_upper > 0:
                        path_pcd_length_lower = path_start_indices[path_number-1]
                        for path_pcd_index in range(path_pcd_length_lower, path_pcd_length_upper):
                            # perform path_stride steps with pcd
                            # predict mesh at t+1 by using grid and collider of step t or t+1
                            data_timestep = get_timestep_data(trajectory_list_eval_raw[traj_i][path_pcd_index], last_prediction_with_pcd, input_timestep, use_color, use_poisson, tissue_task)
                            current_edge_radius_dict = get_radius_dict_for_evaluation_mode(0, edge_radius_dict, mode_pcd, 0)
                            data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates,
                                                         hetero=hetero, tissue_task=tissue_task)
                            data = transform_position_to_edges(data, euclidian_distance)
                            data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                            data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                            data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                            data.to(device)

                            # predict next mesh
                            velocity, h_pcd, c_pcd = model(data, h_pcd, c_pcd)
                            old_position = last_prediction_with_pcd.to(device)
                            predicted_position = old_position + velocity.squeeze()
                            last_prediction_with_pcd = predicted_position
                            pcd_loss = criterion(predicted_position, data.y)
                            path_pcd_loss += (pcd_loss.item()/loss_normalizer)
                            path_pcd_iou += calculate_mesh_iou(data.y.cpu().numpy(), predicted_position.detach().cpu().numpy(), mesh_triangles, mesh_triangles)

                        # Loss for current path
                        path_loss += path_pcd_loss
                        path_iou += path_pcd_iou
                        h_0 = h_pcd
                        c_0 = c_pcd

                    # perform path_mesh_length steps with mesh only
                    for path_mesh_index in range(path_pcd_length_upper, path_pcd_length_upper + path_mesh_length):
                        data_timestep = get_timestep_data(trajectory_list_eval_raw[traj_i][path_mesh_index], predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                        current_edge_radius_dict = get_radius_dict_for_evaluation_mode(0, edge_radius_dict, mode_mesh, 0)
                        data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates,
                                                     hetero=hetero, tissue_task=tissue_task)
                        data = transform_position_to_edges(data, euclidian_distance)
                        data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                        data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                        data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                        data.to(device)

                        # predict next mesh
                        velocity, h_0, c_0 = model(data, h_0, c_0)
                        old_position = predicted_position.to(device)
                        predicted_position = old_position + velocity.squeeze()
                        mesh_loss = criterion(predicted_position, data.y)
                        path_loss += (mesh_loss.item()/loss_normalizer)
                        path_iou += calculate_mesh_iou(data.y.cpu().numpy(), predicted_position.detach().cpu().numpy(), mesh_triangles, mesh_triangles)

                    # loss for whole trajectory
                    traj_loss += path_loss
                    traj_iou += path_iou
                traj_loss = traj_loss / np.sum(path_start_indices + path_mesh_length)
                traj_iou = traj_iou / np.sum(path_start_indices + path_mesh_length)
                eval_loss += traj_loss
                eval_iou += traj_iou
                pbar.update(1)
        pbar.close()
        eval_loss = eval_loss.item()/len(trajectory_list_eval_raw)
        eval_iou = eval_iou/len(trajectory_list_eval_raw)
        print(f'MPC eval {path_mesh_length}-mesh loss: {eval_loss}')

    return eval_loss, eval_iou


def mpc_m_mesh_evaluation(model,
                      hetero: bool,
                      path_mesh_length: int,
                      path_stride: int,
                      trajectory_list_eval_raw: list,
                      device,
                      epoch: int,
                      loss_normalizer: float,
                      edge_radius_dict: Dict,
                      input_timestep="t+1",
                      mode_pcd="full-pcd",
                      mode_mesh="mesh-only",
                      euclidian_distance=False,
                      use_color=False,
                      use_poisson=False,
                      tissue_task=False,
                      use_world_edges=False,
                      use_mesh_coordinates=False,
                      mgn_hetero=False) -> float:
    """
    Function to evaluate our model on T-step predictions with T <= len(traj). path_mesh_length steps include no point cloud.
    Args:
        model: The torch.nn.Module model to evaluate
        hetero: True if heterogeneous data is used
        path_mesh_length: Number of time steps without point cloud but mesh only
        path_stride: Stride for T-step predictions
        trajectory_list_eval_raw: The Dataloader for the evaluation data
        device: Working device, either cpu or cuda
        epoch: Current epoch
        loss_normalizer: Normalizer for the loss
        edge_radius_dict: Dictionary containing the edge radii for the used connectivity setting
        input_timestep: Either 't' or 't+1' if point cloud and collider of future time step is used
        mode_pcd: Evaluation mode for time steps with point cloud
        mode_mesh: Evaluation mode for time steps without point cloud
        euclidian_distance: True if Euclidean norm included as feature
        use_color: Color gradient texture for point cloud
        use_poisson: True if the poisson's ratio is used as feature
        tissue_task: True if 3D data is used
        use_world_edges: Use of explicit world edges (see MGN)
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        mgn_hetero: Use small hetero GNN like in MGN
    Returns:
        eval_loss: Mean MSE loss over the predictions of all time steps in all trajectories
    """
    model.eval()
    criterion = nn.MSELoss()
    pbar = tqdm(total=len(trajectory_list_eval_raw))
    pbar.set_description(f'MPC eval {path_mesh_length}-mesh loss: {epoch:04d}')
    len_traj = len(trajectory_list_eval_raw[0])
    max_traj_index = len_traj - path_mesh_length
    path_start_indices = np.arange(0, max_traj_index + 1, path_stride)
    num_paths = len(path_start_indices)
    eval_loss = 0

    with torch.no_grad():
        for traj_i in range(len(trajectory_list_eval_raw)):
            traj_loss = 0
            path_pcd_loss = 0
            h_0 = None
            c_0 = None
            h_pcd = h_0
            c_pcd = h_0
            predicted_position = trajectory_list_eval_raw[traj_i][0].y_old
            last_prediction_with_pcd = predicted_position

            for path_number in range(num_paths):
                path_loss = 0
                path_pcd_length_upper = path_start_indices[path_number]

                if path_pcd_length_upper > 0:
                    path_pcd_length_lower = path_start_indices[path_number-1]
                    for path_pcd_index in range(path_pcd_length_lower, path_pcd_length_upper):
                        # perform path_stride steps with pcd
                        # predict mesh at t+1 by using grid and collider of step t or t+1
                        data_timestep = get_timestep_data(trajectory_list_eval_raw[traj_i][path_pcd_index], last_prediction_with_pcd, input_timestep, use_color, use_poisson, tissue_task)
                        current_edge_radius_dict = get_radius_dict_for_evaluation_mode(0, edge_radius_dict, mode_pcd, 0)
                        data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates,
                                                     hetero=hetero, tissue_task=tissue_task)
                        data = transform_position_to_edges(data, euclidian_distance)
                        data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                        data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                        data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                        data.to(device)

                        # predict next mesh
                        velocity, h_pcd, c_pcd = model(data, h_pcd, c_pcd)
                        old_position = last_prediction_with_pcd.to(device)
                        predicted_position = old_position + velocity.squeeze()
                        last_prediction_with_pcd = predicted_position
                        pcd_loss = criterion(predicted_position, data.y)
                        path_pcd_loss += (pcd_loss.item()/loss_normalizer)

                    # Loss for current path
                    path_loss += path_pcd_loss
                    h_0 = h_pcd
                    c_0 = c_pcd

                # perform path_mesh_length steps with mesh only
                for path_mesh_index in range(path_pcd_length_upper, path_pcd_length_upper + path_mesh_length):
                    data_timestep = get_timestep_data(trajectory_list_eval_raw[traj_i][path_mesh_index], predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                    current_edge_radius_dict = get_radius_dict_for_evaluation_mode(0, edge_radius_dict, mode_mesh, 0)
                    data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates,
                                                 hetero=hetero, tissue_task=tissue_task)
                    data = transform_position_to_edges(data, euclidian_distance)
                    data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                    data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                    data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                    data.to(device)

                    # predict next mesh
                    velocity, h_0, c_0 = model(data, h_0, c_0)
                    old_position = predicted_position.to(device)
                    predicted_position = old_position + velocity.squeeze()
                    mesh_loss = criterion(predicted_position, data.y)
                    path_loss += (mesh_loss.item()/loss_normalizer)

                # loss for whole trajectory
                traj_loss += path_loss
            traj_loss = traj_loss / np.sum(path_start_indices + path_mesh_length)
            eval_loss += traj_loss
            pbar.update(1)
    pbar.close()
    eval_loss = eval_loss.item()/len(trajectory_list_eval_raw)
    print(f'MPC eval {path_mesh_length}-mesh loss: {eval_loss}')

    return eval_loss


def test_model(GNN,
               input_dataset: str,
               directory: str,
               run_folder_dir: str,
               name: str,
               edge_radius_dict: Dict,
               input_timestep: str,
               euclidian_distance: bool,
               hetero: bool,
               use_color: bool,
               use_poisson: bool,
               tissue_task: bool,
               batch_size: int,
               device,
               loss_normalizer: float,
               additional_eval: bool,
               use_world_edges: bool,
               use_mesh_coordinates: bool,
               mgn_hetero: bool,
               sequence_length: int):
    """

    Args:
        GNN: GNN model to test
        input_dataset: Name of the Dataset
        directory: Current working directory
        run_folder_dir: Folder of the current wandb run
        name: Name of test method
        edge_radius_dict:
        input_timestep:
        euclidian_distance:
        hetero:
        use_color:
        use_poisson:
        tissue_task:
        batch_size:
        device:
        loss_normalizer:
        additional_eval:
        use_world_edges:
        use_mesh_coordinates:
        mgn_hetero:
    """
    # define the additional test scenarios
    test_mode_list = ['k-hop', 'k-hop', 'start-plus-k', 'start-plus-k', 'mpc10']
    k_list = [2, 5, 5, 10, 0]
    test_mode_name_list = ['complete rollout/2-hop',
                           'complete rollout/5-hop',
                           'complete rollout/start-plus-5',
                           'complete rollout/start-plus-10',
                           'imputation mpc/10-mesh']
    m_list = [0, 0, 0, 0, 10]
    save_parameters_list = [True, True, False, False, True]

    # get triangles of mesh for iou
    triangles_grid, _ = get_mesh_triangles_from_sofa(input_dataset, directory, name, tissue_task=tissue_task)


    # generate test data
    trajectory_list_test = build_dataset_for_split(input_dataset, directory, name, edge_radius_dict, 'cpu', input_timestep, use_mesh_coordinates, hetero=hetero, use_color=use_color, use_poisson=use_poisson, tissue_task=tissue_task)
    trajectory_list_test_raw = build_dataset_for_split(input_dataset, directory, name, edge_radius_dict, 'cpu', input_timestep, use_mesh_coordinates, hetero=hetero, raw=True, use_color=use_color, use_poisson=use_poisson, tissue_task=tissue_task)
    test_data = SequenceNoReturnDataset(trajectory_list_test, sequence_length)
    testloader_single = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    shortest_test_trajectory = get_shortest_trajectory(trajectory_list_test)

    # load best evaluation parameters and perform tests
    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_single_step"), map_location=device))
    test_loss_single_step = single_step_evaluation(GNN, hetero, testloader_single, device, 0, loss_normalizer, euclidian_distance, use_color, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
    _, test_iou_single_step = n_step_evaluation_iou(GNN, hetero, 1, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid)
    wandb.run.summary[name.replace("_", " ") + " loss/single step"] = test_loss_single_step
    wandb.run.summary[name.replace("_", " ") + " IOU/single step"] = test_iou_single_step
    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_10_step"), map_location=device))
    test_loss_10_step, test_iou_10_step = n_step_evaluation_iou(GNN, hetero, 10, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid)
    wandb.run.summary[name.replace("_", " ") + " loss/10 step"] = test_loss_10_step
    wandb.run.summary[name.replace("_", " ") + " IOU/10 step"] = test_iou_10_step
    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_complete_rollout"), map_location=device))
    test_loss_complete_rollout, test_iou_complete_rollout = n_step_evaluation_iou(GNN, hetero, shortest_test_trajectory, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid)
    wandb.run.summary[name.replace("_", " ") + " loss/complete rollout"] = test_loss_complete_rollout
    wandb.run.summary[name.replace("_", " ") + " IOU/complete rollout"] = test_iou_complete_rollout

    current_test_loss_complete_rollout, current_test_iou_complete_rollout = mpc_m_mesh_evaluation_iou(GNN, hetero, 10, 1, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 'full-pcd', 'full-pcd', euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid)
    wandb.run.summary[name.replace("_", " ") + " loss mpc/10-mesh"] = current_test_loss_complete_rollout
    wandb.run.summary[name.replace("_", " ") + " IOU mpc/10-mesh"] = current_test_iou_complete_rollout

    # tests on the additional scenarios
    if additional_eval:
        for i, mode in enumerate(test_mode_list):
            m = m_list[i]
            if m == 0:
                if save_parameters_list[i]:
                    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_" + test_mode_name_list[i].replace(" ", "_").replace("/", "_")), map_location=device))
                else:
                    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_complete_rollout"), map_location=device))
                current_test_loss_complete_rollout, current_test_iou_complete_rollout = n_step_evaluation_iou(GNN, hetero, shortest_test_trajectory, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, mode, k_list[i], euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid)
                wandb.run.summary[name.replace("_", " ") + " loss " + test_mode_name_list[i]] = current_test_loss_complete_rollout
                wandb.run.summary[name.replace("_", " ") + " IOU " + test_mode_name_list[i]] = current_test_iou_complete_rollout
            else:
                if save_parameters_list[i]:
                    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_" + test_mode_name_list[i].replace(" ", "_").replace("/", "_")), map_location=device))
                else:
                    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_complete_rollout"), map_location=device))
                current_test_loss_complete_rollout, current_test_iou_complete_rollout = mpc_m_mesh_evaluation_iou(GNN, hetero, m, 1, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 'full-pcd', 'mesh-only', euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid)
                wandb.run.summary[name.replace("_", " ") + " loss " + test_mode_name_list[i]] = current_test_loss_complete_rollout
                wandb.run.summary[name.replace("_", " ") + " IOU " + test_mode_name_list[i]] = current_test_iou_complete_rollout


def train(config: Dict, wandb_log: bool):
    """
    Main training script for Observation-aided Physics Simulation using Graph Neural Networks without imputation (global GNN)
    Functionalities: Loads the data, initializes the GNN and it utils, trains the model and logs the losses
    Args:
        config: config dictionary loaded from a .yaml config file
        wandb_log: (bool) use wandb logging or not
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print("Using device: ", device)
    ROOT_DIR = os.path.dirname(os.path.abspath('main.py'))
    config, run_folder_dir = wandb_init(config, wandb_log, ROOT_DIR)

    # general parameters
    hetero = bool(config.get("hetero"))
    mgn_hetero = bool(config.get("mgn_hetero"))
    use_poisson = bool(config.get("use_poisson"))

    # dataset parameters
    dataset = config.get("connectivity_setting")
    input_dataset = config.get("build_from")
    use_color = bool(config.get("use_colors"))
    use_mesh_coordinates = bool(config.get("use_mesh_coordinates"))
    use_world_edges = True if "world" in dataset or mgn_hetero else False

    # hyperparameters
    batch_size = config.get("batch_size")
    loss_normalizer = config.get("loss_normalizer")
    learning_rate = config.get("learning_rate")
    weight_decay = config.get("weight_decay")
    input_mesh_noise = config.get("input_mesh_noise")
    final_test = config.get("final_test")
    clip_gradient_norm = config.get("clip_gradient_norm")
    num_epochs = config.get("num_epochs")
    eval_log_interval = config.get("eval_log_interval")
    seed_run = config.get("run_num")

    # Imputation parameters
    additional_eval = config.get("additional_eval")
    # Point cloud dropout parameters
    pointcloud_dropout = config.get("pointcloud_dropout")
    # Weighted Dataset parameters
    weighted_dataset = bool(config.get("weighted_dataset"))
    sequence_length = config.get("sequence_length")
    pcd_weighting = None if "None" in str(config.get("pcd_weighting")) else config.get("pcd_weighting")

    # set seed
    seed_all(seed_run)

    # get parameters for GNNBase and dataset
    network_config = get_network_config(config, hetero, use_world_edges)
    edge_radius_dict, input_timestep, euclidian_distance, tissue_task = get_connectivity_setting(dataset)

    # build dataset from sofa data
    trajectory_list_train = build_dataset_for_split(input_dataset, ROOT_DIR, "train", edge_radius_dict, 'cpu', input_timestep, use_mesh_coordinates, hetero=hetero, use_color=use_color, use_poisson=use_poisson, tissue_task=tissue_task)
    trajectory_list_eval = build_dataset_for_split(input_dataset, ROOT_DIR, "eval", edge_radius_dict, 'cpu', input_timestep, use_mesh_coordinates, hetero=hetero, use_color=use_color, use_poisson=use_poisson, tissue_task=tissue_task)
    trajectory_list_eval_raw = build_dataset_for_split(input_dataset, ROOT_DIR, "eval", edge_radius_dict, 'cpu', input_timestep, use_mesh_coordinates, hetero=hetero, raw=True, use_color=use_color, use_poisson=use_poisson, tissue_task=tissue_task)

    # data information for building GNN
    data_list = trajectory_list_train[0]
    data_point = transform_position_to_edges(copy.deepcopy(data_list[0]).to(device), euclidian_distance)
    in_node_features, in_edge_features, num_node_features, num_edge_features = get_feature_info_from_data(data_point, device, hetero, use_color, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
    out_node_features = data_point.y.shape[1]  # 2 for 2D case
    in_global_features = 1
    len_trajectory = len(data_list)
    shortest_eval_trajectory = get_shortest_trajectory(trajectory_list_eval)
    del data_list
    del data_point
    wandb_config_update(wandb_log, {"num_node_features": num_node_features, "num_edge_features": num_edge_features})

    # build GNN model
    GNN = get_gnn_model(in_node_features, in_edge_features, in_global_features, out_node_features, network_config, hetero, device)
    num_params = count_parameters(GNN, show_details=False)
    #print([module for module in GNN.modules() if not isinstance(module, nn.Sequential)][0])
    wandb_config_update(wandb_log, {"num_params": num_params})

    # prepare train data
    if weighted_dataset:
        if tissue_task:
            if "tube" in dataset:
                edge_radius_dict_mgn, input_timestep_mgn, euclidian_distance_mgn, _ = get_connectivity_setting("tissue_tube_meshgraphnet")
            else:
                edge_radius_dict_mgn, input_timestep_mgn, euclidian_distance_mgn, _ = get_connectivity_setting("tissue_norm_meshgraphnet_t")
        else:
            edge_radius_dict_mgn, input_timestep_mgn, euclidian_distance_mgn, _ = get_connectivity_setting("coarse_meshgraphnet_t")
        trajectory_list_train_mgn = build_dataset_for_split(input_dataset, ROOT_DIR, "train", edge_radius_dict_mgn, 'cpu', input_timestep_mgn, use_mesh_coordinates, hetero=hetero, use_color=use_color, use_poisson=use_poisson, tissue_task=tissue_task)
        trajectory_list_train_mgn = crop_list_of_trajectories(trajectory_list_train_mgn, 0, len_trajectory)
        trajectory_list_train = crop_list_of_trajectories(trajectory_list_train, 0, len_trajectory)
        train_dataset = WeightedSequenceNoReturnDataset(trajectory_list_train, trajectory_list_train_mgn, sequence_length)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    else:
        train_dataset = SequenceNoReturnDataset(trajectory_list_train, sequence_length)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker)

    # prepare eval data
    eval_dataset = SequenceNoReturnDataset(trajectory_list_eval, sequence_length)
    evalloader_single = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker)


    if loss_normalizer == 'None' or loss_normalizer is None:
        loss_normalizer = calculate_loss_normalizer(trainloader, device)

    # initialize best losses
    best_train_loss = float("inf")
    best_eval_loss_single_step = float("inf")
    best_eval_loss_10_step = float("inf")
    best_eval_loss_complete_rollout = float("inf")
    if additional_eval:
        eval_mode_list = ['k-hop', 'k-hop', 'mpc10']
        eval_mode_name_list = ['eval loss complete rollout/2-hop',
                               'eval loss complete rollout/5-hop',
                               'eval loss imputation mpc/10-mesh']
        k_list = [2, 5, 0]
        m_list = [0, 0, 10]
        additional_eval_list = [float("inf"), float("inf"), float("inf"), float("inf")]

    # main training loop
    optimizer = torch.optim.Adam(GNN.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    for epoch in range(1, num_epochs + 1):
        GNN.train()
        pbar = tqdm(total=len(trainloader))
        pbar.set_description(f'Training epoch: {epoch:04d}')
        total_loss = 0
        for data_list in trainloader:
            optimizer.zero_grad()
            target = []
            old_position = []
            node_features_list = []
            h_0 = None
            c_0 = None
            for data in data_list:
                data = add_pointcloud_dropout(data, pointcloud_dropout, hetero, use_world_edges)
                data.to(device)
                data = add_noise_to_mesh_nodes(data, input_mesh_noise, device)
                data = transform_position_to_edges(data, euclidian_distance)
                data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                node_features_out, h_0, c_0 = GNN(data, h_0, c_0)
                node_features_list.append(node_features_out)
                target.append(data.y)
                old_position.append(data.y_old)
            target = torch.stack(target, dim=1)
            old_position = torch.stack(old_position, dim=1)
            node_features = torch.cat(node_features_list, dim=1)
            predicted_position = old_position + node_features
            loss = criterion(predicted_position, target)
            loss.backward()
            if clip_gradient_norm > 0:
                nn.utils.clip_grad_norm_(GNN.parameters(), clip_gradient_norm)
            optimizer.step()
            total_loss += loss
            pbar.update(1)
        pbar.close()
        total_loss = total_loss.item()/len(trainloader)/loss_normalizer
        print("Loss epoch ", epoch, ": ", total_loss)
        best_train_loss = wandb_loss_logger(wandb_log, total_loss, "train loss", epoch, best_train_loss, GNN, run_folder_dir)
        log_gradients(GNN, wandb_log, epoch, eval_log_interval, hetero)

        if epoch % eval_log_interval == 0:
            eval_loss_single_step = single_step_evaluation(GNN, hetero, evalloader_single, device, epoch, loss_normalizer, euclidian_distance, use_color, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
            eval_loss_10_step = n_step_evaluation(GNN, hetero, 10, trajectory_list_eval_raw, device, epoch, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
            eval_loss_complete_rollout = n_step_evaluation(GNN, hetero, shortest_eval_trajectory, trajectory_list_eval_raw, device, epoch, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)

            # log evaluation losses using wandb
            best_eval_loss_single_step = wandb_loss_logger(wandb_log, eval_loss_single_step, "eval loss/single step", epoch, best_eval_loss_single_step, GNN, run_folder_dir)
            best_eval_loss_10_step = wandb_loss_logger(wandb_log, eval_loss_10_step, "eval loss/10 step", epoch, best_eval_loss_10_step, GNN, run_folder_dir)
            best_eval_loss_complete_rollout = wandb_loss_logger(wandb_log, eval_loss_complete_rollout, "eval loss/complete rollout", epoch, best_eval_loss_complete_rollout, GNN, run_folder_dir)

            if additional_eval:
                for i, mode in enumerate(eval_mode_list):
                    m = m_list[i]
                    if m == 0:
                        current_eval_loss_complete_rollout = n_step_evaluation(GNN, hetero, shortest_eval_trajectory, trajectory_list_eval_raw, device, epoch, loss_normalizer, edge_radius_dict, input_timestep, mode, k_list[i], euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                        additional_eval_list[i] = wandb_loss_logger(wandb_log, current_eval_loss_complete_rollout, eval_mode_name_list[i], epoch, additional_eval_list[i], GNN, run_folder_dir)
                    else:
                        if tissue_task:
                            mpc_hop = 20
                        else:
                            mpc_hop = 10
                        current_eval_loss = mpc_m_mesh_evaluation(GNN, hetero, m, mpc_hop, trajectory_list_eval_raw, device, epoch, loss_normalizer, edge_radius_dict, input_timestep, 'full-pcd', 'mesh-only', euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                        additional_eval_list[i] = wandb_loss_logger(wandb_log, current_eval_loss, eval_mode_name_list[i], epoch, additional_eval_list[i], GNN, run_folder_dir)

    if wandb_log:
        if final_test:
            test_model(GNN, input_dataset, ROOT_DIR, run_folder_dir, "test", edge_radius_dict, input_timestep, euclidian_distance, hetero, use_color, use_poisson, tissue_task, batch_size, device, loss_normalizer, additional_eval, use_world_edges, use_mesh_coordinates, mgn_hetero, sequence_length)

    wandb.finish()
