import os
import numpy as np
import pymesh
import torch
from torch import nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb

from src.utils.dataset_utils import build_dataset_for_split, get_mesh_triangles_from_sofa
from src.utils.train_utils import add_noise_to_pcd_points
from src.utils.get_connectivity_setting import get_radius_dict
from src.utils.data_utils import get_timestep_data, convert_to_hetero_data, get_shortest_trajectory, \
    convert_trajectory_to_data_list, predict_velocity, transform_position_to_edges
from src.utils.graph_utils import create_graph_from_raw
from util.Types import *


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
                           mgn_hetero=False,
                           input_pcd_noise=0.0) -> float:
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
        for data in evalloader:
            data.to(device)
            data = add_noise_to_pcd_points(data, input_pcd_noise, device)
            data = transform_position_to_edges(data, euclidian_distance)
            data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
            node_features_out, _, _ = model(data)
            velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
            predicted_position = data.y_old + velocity
            criterion = nn.MSELoss()
            loss = criterion(predicted_position, data.y)
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
                      mesh_triangles=None,
                      input_pcd_noise=0.0) -> tuple:
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
                      mgn_hetero,
                      input_pcd_noise=0.0)
        eval_iou = 0.0
    else:
        model.eval()
        pbar = tqdm(total=len(trajectory_list_eval_raw))
        pbar.set_description(f'{n}_step eval loss: {epoch:04d}')
        eval_loss = 0
        eval_iou = 0
        with torch.no_grad():
            for data_list in trajectory_list_eval_raw:
                for index, data in enumerate(data_list):

                    # inputs the ground truth mesh after n timesteps
                    if index % n == 0:
                        predicted_position = data.y_old.to(device)

                    # get correct data for current timestep from raw data and build graph
                    data_timestep = get_timestep_data(data, predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                    current_edge_radius_dict = get_radius_dict_for_evaluation_mode(index, edge_radius_dict, mode, k)
                    data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates, hetero=hetero, tissue_task=tissue_task)
                    data = add_noise_to_pcd_points(data, input_pcd_noise, device)
                    data = transform_position_to_edges(data, euclidian_distance)
                    data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                    data.ptr = torch.tensor([0, data.batch.shape[0]])  # needed if mgn_hetero is used
                    data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                    data.to(device)

                    # evaluate model
                    node_features_out, _, _ = model(data)
                    old_position = predicted_position
                    velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
                    predicted_position = old_position + velocity
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
                      mgn_hetero=False,
                      input_pcd_noise=0.0) -> float:
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
            for index, data in enumerate(data_list):

                # inputs the ground truth mesh after n timesteps
                if index % n == 0:
                    predicted_position = data.y_old.to(device)

                # get correct data for current timestep from raw data and build graph
                data_timestep = get_timestep_data(data, predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                current_edge_radius_dict = get_radius_dict_for_evaluation_mode(index, edge_radius_dict, mode, k)
                data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates, hetero=hetero, tissue_task=tissue_task)
                data = add_noise_to_pcd_points(data, input_pcd_noise, device)
                data = transform_position_to_edges(data, euclidian_distance)
                data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                data.to(device)

                # evaluate model
                node_features_out, _, _ = model(data)
                old_position = predicted_position
                velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
                predicted_position = old_position + velocity
                criterion = nn.MSELoss()
                loss = criterion(predicted_position, data.y)
                eval_loss += loss
            pbar.update(1)
    pbar.close()
    eval_loss = eval_loss.item()/len(trajectory_list_eval_raw)/len(data_list)/loss_normalizer
    print(str(n) + "_step eval loss: ", eval_loss)

    return eval_loss


def n_plus_m_step_evaluation(model,
                      hetero: bool,
                      n: int,
                      m: int,
                      trajectory_list_eval_raw: list,
                      device,
                      epoch: int,
                      loss_normalizer: float,
                      edge_radius_dict: Dict,
                      input_timestep="t+1",
                      mode="start-plus-k",
                      k=5,
                      euclidian_distance=False,
                      use_color=False,
                      use_poisson=False,
                      tissue_task=False,
                      use_world_edges=False,
                      use_mesh_coordinates=False,
                      mgn_hetero=False,
                      input_pcd_noise=0.0) -> float:
    """
    Function to evaluate our model on n+m-step predictions. This means the ground truth mesh is used as input in every n-th time step.
    A point cloud is used in the first k steps (depending on mode) and subsequently the model is rolled out for m steps.
    This is done for all possible shifts of the start point in the trajectory by k.
    Args:
        model: The torch.nn.Module model to evaluate
        hetero: True if heterogeneous data is used
        n: Number of time steps after which a ground truth mesh is input again
        m: Number of time steps to rollout
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
    pbar.set_description(f'{k}_plus_{m}_{k+m}_step eval loss: {epoch:04d}')
    len_traj = len(trajectory_list_eval_raw[0])
    max_traj_index = len_traj - k - m
    num_traj_index = int(max_traj_index/k) + 1
    eval_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for data_list in trajectory_list_eval_raw:
            traj_loss = 0
            for traj_index in range(num_traj_index):
                current_data_list = data_list[traj_index*k:(traj_index+1)*k + m]
                for index, data in enumerate(current_data_list):

                    # inputs the ground truth mesh after n timesteps
                    if index % n == 0:
                        predicted_position = data.y_old.to(device)

                    # get correct data for current timestep from raw data and build graph
                    data_timestep = get_timestep_data(data, predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                    current_edge_radius_dict = get_radius_dict_for_evaluation_mode(index, edge_radius_dict, mode, k)
                    data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates, hetero=hetero, tissue_task=tissue_task)
                    data = add_noise_to_pcd_points(data, input_pcd_noise, device)
                    data = transform_position_to_edges(data, euclidian_distance)
                    data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                    data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                    data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                    data.to(device)

                    # evaluate model
                    node_features_out, _, _ = model(data)
                    old_position = predicted_position
                    velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
                    predicted_position = old_position + velocity
                    loss = criterion(predicted_position, data.y)
                    traj_loss += loss
            pbar.update(1)
            traj_loss = traj_loss / num_traj_index / (k + m)
            eval_loss += traj_loss
    pbar.close()
    eval_loss = eval_loss.item()/len(trajectory_list_eval_raw)/loss_normalizer
    print(f'{k}_plus_{m}_{k+m}_step eval loss: {eval_loss}')

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
                      mesh_triangles=None,
                      input_pcd_noise=0.0) -> tuple:
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
                      mgn_hetero,
                      input_pcd_noise)
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
                            data = add_noise_to_pcd_points(data, input_pcd_noise, device)
                            data = transform_position_to_edges(data, euclidian_distance)
                            data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                            data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                            data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                            data.to(device)

                            # predict next mesh
                            node_features_out, _, _ = model(data)
                            old_position = last_prediction_with_pcd.to(device)
                            velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
                            predicted_position = old_position + velocity
                            last_prediction_with_pcd = predicted_position
                            pcd_loss = criterion(predicted_position, data.y)
                            path_pcd_loss += (pcd_loss.item()/loss_normalizer)
                            path_pcd_iou += calculate_mesh_iou(data.y.cpu().numpy(), predicted_position.detach().cpu().numpy(), mesh_triangles, mesh_triangles)

                        # Loss for current path
                        path_loss += path_pcd_loss
                        path_iou += path_pcd_iou

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
                        node_features_out, _, _ = model(data)
                        old_position = predicted_position.to(device)
                        velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
                        predicted_position = old_position + velocity
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
                      mgn_hetero=False,
                      input_pcd_noise=0.0) -> float:
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
                        data = add_noise_to_pcd_points(data, input_pcd_noise, device)
                        data = transform_position_to_edges(data, euclidian_distance)
                        data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                        data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                        data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                        data.to(device)

                        # predict next mesh
                        node_features_out, _, _ = model(data)
                        old_position = last_prediction_with_pcd.to(device)
                        velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
                        predicted_position = old_position + velocity
                        last_prediction_with_pcd = predicted_position
                        pcd_loss = criterion(predicted_position, data.y)
                        path_pcd_loss += (pcd_loss.item()/loss_normalizer)

                    # Loss for current path
                    path_loss += path_pcd_loss

                # perform path_mesh_length steps with mesh only
                for path_mesh_index in range(path_pcd_length_upper, path_pcd_length_upper + path_mesh_length):
                    data_timestep = get_timestep_data(trajectory_list_eval_raw[traj_i][path_mesh_index], predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                    current_edge_radius_dict = get_radius_dict_for_evaluation_mode(0, edge_radius_dict, mode_mesh, 0)
                    data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates,
                                                 hetero=hetero, tissue_task=tissue_task)
                    data = add_noise_to_pcd_points(data, input_pcd_noise, device)
                    data = transform_position_to_edges(data, euclidian_distance)
                    data.batch = torch.zeros_like(data.x).long()  # needed if mgn_hetero is used
                    data.ptr = torch.tensor([0, data.batch.shape[0]]) # needed if mgn_hetero is used
                    data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                    data.to(device)

                    # predict next mesh
                    node_features_out, _, _ = model(data)
                    old_position = predicted_position.to(device)
                    velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
                    predicted_position = old_position + velocity
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
               input_pcd_noise=0.0):
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
    # test_mode_list = ['mesh-only', 'k-hop', 'k-hop', 'k-hop', 'start-plus-k', 'start-plus-k', 'start-plus-k', 'start-plus-k', 'start-plus-k']
    # k_list = [0, 2, 5, 10, 20, 10, 5, 5, 5]
    # test_mode_name_list = ['loss complete rollout/mesh-only',
    #                            'loss complete rollout/2-hop',
    #                            'loss complete rollout/5-hop',
    #                            'loss complete rollout/10-hop',
    #                            'loss complete rollout/start-plus-20',
    #                            'loss complete rollout/start-plus-10',
    #                            'loss complete rollout/start-plus-5',
    #                            'loss 5-plus-10 step/start-plus-5',
    #                            'loss 5-plus-20 step/start-plus-5']
    # m_list = [0, 0, 0, 0, 0, 0, 0, 10, 20]
    test_mode_list = ['k-hop', 'k-hop', 'k-hop', 'start-plus-k', 'start-plus-k', 'mpc10']
    k_list = [2, 5, 10, 5, 10, 0]
    test_mode_name_list = ['complete rollout/2-hop',
                           'complete rollout/5-hop',
                           'complete rollout/10-hop',
                           'complete rollout/start-plus-5',
                           'complete rollout/start-plus-10',
                           'imputation mpc/10-mesh']
    m_list = [0, 0, 0, 0, 0, 10]
    save_parameters_list = [True, True, True, False, False, True]

    # get triangles of mesh for iou
    triangles_grid, _ = get_mesh_triangles_from_sofa(input_dataset, directory, name, tissue_task=tissue_task)


    # generate test data
    trajectory_list_test = build_dataset_for_split(input_dataset, directory, name, edge_radius_dict, 'cpu', input_timestep, use_mesh_coordinates, hetero=hetero, use_color=use_color, use_poisson=use_poisson, tissue_task=tissue_task)
    trajectory_list_test_raw = build_dataset_for_split(input_dataset, directory, name, edge_radius_dict, 'cpu', input_timestep, use_mesh_coordinates, hetero=hetero, raw=True, use_color=use_color, use_poisson=use_poisson, tissue_task=tissue_task)
    test_data_list = convert_trajectory_to_data_list(trajectory_list_test)
    testloader_single = DataLoader(test_data_list, batch_size=batch_size, shuffle=True, pin_memory=True)
    shortest_test_trajectory = get_shortest_trajectory(trajectory_list_test)

    # load best evaluation parameters and perform tests
    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_single_step"), map_location=device))
    test_loss_single_step = single_step_evaluation(GNN, hetero, testloader_single, device, 0, loss_normalizer, euclidian_distance, use_color, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)
    _, test_iou_single_step = n_step_evaluation_iou(GNN, hetero, 1, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid, input_pcd_noise)
    wandb.run.summary[name.replace("_", " ") + " loss/single step"] = test_loss_single_step
    wandb.run.summary[name.replace("_", " ") + " IOU/single step"] = test_iou_single_step
    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_10_step"), map_location=device))
    test_loss_10_step, test_iou_10_step = n_step_evaluation_iou(GNN, hetero, 10, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid, input_pcd_noise)
    wandb.run.summary[name.replace("_", " ") + " loss/10 step"] = test_loss_10_step
    wandb.run.summary[name.replace("_", " ") + " IOU/10 step"] = test_iou_10_step
    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_complete_rollout"), map_location=device))
    test_loss_complete_rollout, test_iou_complete_rollout = n_step_evaluation_iou(GNN, hetero, shortest_test_trajectory, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid, input_pcd_noise)
    wandb.run.summary[name.replace("_", " ") + " loss/complete rollout"] = test_loss_complete_rollout
    wandb.run.summary[name.replace("_", " ") + " IOU/complete rollout"] = test_iou_complete_rollout
    # if not tissue_task:
    #     current_test_loss_complete_rollout, current_test_iou_complete_rollout = mpc_m_mesh_evaluation_iou(GNN, hetero, 5, 1, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 'full-pcd', 'full-pcd', euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid)
    #     wandb.run.summary[name.replace("_", " ") + " loss mpc/5-mesh"] = current_test_loss_complete_rollout
    #     wandb.run.summary[name.replace("_", " ") + " IOU mpc/5-mesh"] = current_test_iou_complete_rollout

    current_test_loss_complete_rollout, current_test_iou_complete_rollout = mpc_m_mesh_evaluation_iou(GNN, hetero, 10, 1, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 'full-pcd', 'full-pcd', euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid, input_pcd_noise)
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
                current_test_loss_complete_rollout, current_test_iou_complete_rollout = n_step_evaluation_iou(GNN, hetero, shortest_test_trajectory, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, mode, k_list[i], euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid, input_pcd_noise)
                wandb.run.summary[name.replace("_", " ") + " loss " + test_mode_name_list[i]] = current_test_loss_complete_rollout
                wandb.run.summary[name.replace("_", " ") + " IOU " + test_mode_name_list[i]] = current_test_iou_complete_rollout
            else:
                if save_parameters_list[i]:
                    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_" + test_mode_name_list[i].replace(" ", "_").replace("/", "_")), map_location=device))
                else:
                    GNN.load_state_dict(torch.load(os.path.join(run_folder_dir, "GNN_model_best_eval_loss_complete_rollout"), map_location=device))
                current_test_loss_complete_rollout, current_test_iou_complete_rollout = mpc_m_mesh_evaluation_iou(GNN, hetero, m, 1, trajectory_list_test_raw, device, 0, loss_normalizer, edge_radius_dict, input_timestep, 'full-pcd', 'mesh-only', euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, triangles_grid, input_pcd_noise)
                wandb.run.summary[name.replace("_", " ") + " loss " + test_mode_name_list[i]] = current_test_loss_complete_rollout
                wandb.run.summary[name.replace("_", " ") + " IOU " + test_mode_name_list[i]] = current_test_iou_complete_rollout


def get_radius_dict_for_evaluation_mode(index: int, edge_radius_dict: Dict, mode: str, k) -> Dict:
    """
    Transforms the edge radius dict for the different evaluation settings:
        mesh-only: Only mesh information is used (and collider)
        0: Use input radius dict (usually containing point clouds)
        k-hop: Input point cloud only each k-th time step
        k-prob: Randomly use point cloud with probability of 1/k in each time step
        initial-plus-k: Input point cloud after 15 until 15+k time steps
        start-plus-k: Input point cloud for the first k time steps
        half-plus-k: Input point cloud after 30 until 30+k time steps
    Args:
        index: Index of the current sample in the trajectory
        edge_radius_dict: Dictionary of the different edge radii
        mode: Evaluation mode
        k: Additional feature for mode

    Returns:
        edge_radius_dict: Transformed dict
    """
    # build mesh radius dict from the corresponding radii from the input radius dict
    mesh_edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mesh_edge_radius[1] = edge_radius_dict['collider', '1', 'collider']
    mesh_edge_radius[5] = edge_radius_dict['mesh', '5', 'collider']
    mesh_edge_radius[8] = edge_radius_dict['collider', '8', 'mesh']
    mesh_edge_radius_dict = get_radius_dict(mesh_edge_radius)

    if mode == "mesh-only":
        edge_radius_dict = mesh_edge_radius_dict
    elif mode == 0:
        edge_radius_dict = edge_radius_dict
    elif mode == "full-pcd":
        edge_radius_dict = edge_radius_dict
    elif mode == "k-hop":
        if index % k == 0:
            edge_radius_dict = edge_radius_dict
        else:
            edge_radius_dict = mesh_edge_radius_dict
    elif mode == "k-prob":
        prop = 1/k
        x = np.random.rand(1)
        if prop > x:
            edge_radius_dict = edge_radius_dict
        else:
            edge_radius_dict = mesh_edge_radius_dict
    elif mode == "initial-plus-k":
        if (index >= 15) & (index < 15+k):
            edge_radius_dict = edge_radius_dict
        else:
            edge_radius_dict = mesh_edge_radius_dict
    elif mode == "start-plus-k":
        if (index >= 0) & (index < k):
            edge_radius_dict = edge_radius_dict
        else:
            edge_radius_dict = mesh_edge_radius_dict
    elif mode == "half-plus-k":
        if (index >= 30) & (index < 30+k):
            edge_radius_dict = edge_radius_dict
        else:
            edge_radius_dict = mesh_edge_radius_dict
    else:
        raise ValueError("mode does not exist")

    return edge_radius_dict


def get_save_dict_value(dict: Dict, key: str, default_value = False):
    """
    Savely gets a value from a dictionary if it does not necessarily exist. If it does not exist it outputs the value instead
    Args:
        dict: Dictionary from wich to get value
        value: Value if dict['key']['value'] does not exist
    Returns:
        feature: value from dict if exist else value
    """
    feature = dict.get(key)
    if feature is None:
        feature = default_value
    else:
        feature = feature['value']
    return feature


def calculate_mesh_iou(target_mesh_nodes, predicted_mesh_nodes, target_triangles, predicted_triangles):
    import pymesh
    target_mesh = pymesh.meshio.form_mesh(target_mesh_nodes, target_triangles)
    predicted_mesh = pymesh.meshio.form_mesh(predicted_mesh_nodes, predicted_triangles)
    intersection_mesh = pymesh.boolean(target_mesh, predicted_mesh, operation="intersection", engine="auto")
    union_mesh = pymesh.boolean(target_mesh, predicted_mesh, operation="union", engine="auto")
    intersection_mesh.add_attribute("face_area")
    union_mesh.add_attribute("face_area")
    intersection_area = np.sum(intersection_mesh.get_attribute("face_area"))
    union_area = np.sum(union_mesh.get_attribute("face_area"))
    iou = intersection_area/union_area

    return iou


def calculate_mesh_iou_queue(target_mesh_nodes, target_triangles, predicted_mesh, queue):
    import pymesh
    target_mesh = pymesh.meshio.form_mesh(target_mesh_nodes, target_triangles)
    intersection_mesh = pymesh.boolean(target_mesh, predicted_mesh, operation="intersection", engine="auto")
    union_mesh = pymesh.boolean(target_mesh, predicted_mesh, operation="union", engine="clipper")
    intersection_mesh.add_attribute("face_area")
    union_mesh.add_attribute("face_area")
    intersection_area = np.sum(intersection_mesh.get_attribute("face_area"))
    union_area = np.sum(union_mesh.get_attribute("face_area"))
    iou = intersection_area/union_area

    ret = queue.get()
    ret['iou'] = iou
    queue.put(ret)
