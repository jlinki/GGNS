import torch
import numpy as np
import copy
import yaml
import torch.nn as nn
from torch.nn import functional as F
import os
import argparse
from util.Types import *

from src.utils.eval_utils import get_radius_dict_for_evaluation_mode, get_save_dict_value, calculate_mesh_iou
from src.utils.graph_utils import create_graph_from_raw
from src.utils.dataset_utils import build_dataset_for_split, get_mesh_triangles_from_sofa
from src.utils.get_connectivity_setting import get_connectivity_setting
from src.utils.data_utils import get_timestep_data, convert_to_hetero_data, get_feature_info_from_data, \
    count_parameters, predict_velocity, transform_position_to_edges
from src.utils.visualization.visualization import plot_loss_curve, plot_velocity_curve
from src.utils.visualization.visualization2d import animate_rollout_mesh_overlay
from src.utils.visualization.visualization3d import animate_rollout3d_tube_mesh_overlay, animate_rollout3d_mesh_side_by_side, animate_rollout3d_scene

from modules.gnn_modules.homogeneous_modules.GNNBase import GNNBase
from modules.gnn_modules.heterogeneous_modules.HeteroGNNBase import HeteroGNNBase


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


def get_network_config(config, hetero, use_world_edges):
    network_config = {"aggregation_function": config.get("aggregation_function")['value'],
                          "latent_dimension": config.get("latent_dimension")['value'],
                            "mlp_decoder": get_save_dict_value(config, 'mlp_decoder'),
                          "base": {
                              "use_global_features": config.get("use_global_features")['value'],
                              "num_blocks": config.get("num_blocks")['value'],
                              "use_residual_connections": config.get("use_residual_connections")['value'],
                              "mlp": {
                                  "activation_function": config.get("activation_function")['value'],
                                  "num_layers": config.get("num_layers")['value'],
                                  #"output_layer": config.get("output_layer")['value'],
                                  "regularization": {
                                      "latent_normalization": config.get("latent_normalization")['value'],
                                      "dropout": config.get("dropout")['value']
                                  }
                              }
                          }
                          }
    if hetero:
        hetero_config = {"het_neighbor_aggregation": config.get("het_neighbor_aggregation")['value'],
                         "het_edge_shared_weights": config.get("het_edge_shared_weights")['value'],
                         "het_node_shared_weights": config.get("het_node_shared_weights")['value'],
                         "het_world_edges": bool(use_world_edges)}
        network_config['base'].update(hetero_config)
    return network_config


def evaluate(args):
    """
    n-step evaluation of our model loaded from the state dict file. Rollouts can be visualized and saved. For all option refer to the parser
    Args:
        args: args from the parser
    """

    # uses cuda if possible
    device = args.device
    detailed_print = args.print
    if device == 'cuda':
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if detailed_print:
        print("Using device: ", device)
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ )))

    # load evaluation options
    run_name = args.run
    start = args.start
    stop = args.stop
    crop_steps = args.crop
    stride = args.stride
    calculate_iou = args.iou
    eval_mode = args.split
    show_rollout = args.show_rollout
    show_curves = args.show_curves
    save_animation = args.save_animation
    n_step = args.n_step
    mode, k = args.mode, args.mode_k
    if args.model_name == None:
        if n_step == 1:
            epoch = "best_eval_loss_single_step"
        else:
            epoch = "best_eval_loss_complete_rollout"
    else:
        epoch = args.model_name

    try:
        with open(os.path.join(ROOT_DIR, "models", run_name, "config.yaml"), 'r') as file:
            try:
                config = yaml.safe_load(file)
                # general parameters
                loss_normalizer = config.get("loss_normalizer")['value']
                dataset = config.get("dataset")['value']
                input_dataset = config.get("build_from")['value']
                # hyperparameters
                hetero = bool(config.get("hetero")['value'])
                use_color = bool(config.get("use_colors")['value'])
                pointcloud_dropout = config.get("pointcloud_dropout")
                mgn_hetero = get_save_dict_value(config, "mgn_hetero")
                use_world_edges = True if "world" in dataset or mgn_hetero else False
                network_config = get_network_config(config, hetero, use_world_edges)
                use_mesh_coordinates = get_save_dict_value(config, "use_mesh_coordinates")
                use_poisson = get_save_dict_value(config, "use_poisson")
            except yaml.YAMLError as exc:
                print(exc)
    except FileNotFoundError as exc:
        print("Using default config")
        raise ValueError
    #use_poisson = False
    loss_normalizer = args.loss_normalizer
    if args.dataset != None:
        input_dataset = args.dataset

    # load data
    edge_radius_dict, input_timestep, euclidian_distance, tissue_task = get_connectivity_setting(dataset)
    trajectory_list_raw = build_dataset_for_split(input_dataset,
                                                  ROOT_DIR,
                                                  eval_mode,
                                                  edge_radius_dict,
                                                  device,
                                                  input_timestep,
                                                  euclidian_distance,
                                                  hetero=hetero,
                                                  raw=True,
                                                  use_color=use_color,
                                                  use_poisson=use_poisson,
                                                  tissue_task=tissue_task)
    trajectory_list_raw = trajectory_list_raw[start:stop]
    triangles_grid, triangles_collider = get_mesh_triangles_from_sofa(input_dataset, ROOT_DIR, eval_mode, tissue_task=tissue_task, output_list=False)

    # data information for building GNN
    data_list = trajectory_list_raw[0]
    data_point = copy.deepcopy(data_list[0])
    data_point = get_timestep_data(data_point, data_point.y_old, input_timestep, use_color, tissue_task=tissue_task, use_poisson=use_poisson)
    data_point = create_graph_from_raw(data_point,
                                       edge_radius_dict=edge_radius_dict,
                                       output_device=device,
                                       use_mesh_coordinates=use_mesh_coordinates,
                                       hetero=hetero,
                                       tissue_task=tissue_task)
    data_point = transform_position_to_edges(data_point, euclidian_distance)
    in_node_features, in_edge_features, num_node_features, num_edge_features = get_feature_info_from_data(data_point, device, hetero, use_color, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
    out_node_features = data_point.y.shape[1]  # 2 for 2D case
    in_global_features = 1
    del data_list
    del data_point

    # build and load GNN
    GNN = get_gnn_model(in_node_features, in_edge_features, in_global_features, out_node_features, network_config, hetero, device)
    if detailed_print:
        count_parameters(GNN, show_details=False)
    GNN.load_state_dict(torch.load(os.path.join(ROOT_DIR, "models", run_name, "GNN_model_" + str(epoch)), map_location=device))
    GNN.eval()

    eval_loss = []
    eval_iou = []
    for i in range(len(trajectory_list_raw)):
        rollout_predicted = []
        rollout_target = []
        rollout_collider = []
        rollout_loss = []
        rollout_iou = []
        rollout_time_difference_loss = []
        mean_mesh_x_velocity = []
        mean_mesh_x_velocity_predicted = []
        mean_mesh_y_velocity = []
        mean_mesh_y_velocity_predicted = []
        mean_mesh_z_velocity = []
        mean_mesh_z_velocity_predicted = []
        h_0 = None
        c_0 = None

        with torch.no_grad():
            for index, data in enumerate(trajectory_list_raw[i][crop_steps:]):
                old_position_true = data.y_old.float()
                collider_position = data.collider_positions
                if index % n_step == 0:
                    predicted_position = trajectory_list_raw[i][crop_steps+index].y_old

                # predict mesh at t+1 by using grid and collider of step t or t+1
                data_timestep = get_timestep_data(data, predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                current_point_cloud = data_timestep[0]
                current_edge_radius_dict = get_radius_dict_for_evaluation_mode(index, edge_radius_dict, mode, k)
                data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates,
                                             hetero=hetero, tissue_task=tissue_task)
                data = transform_position_to_edges(data, euclidian_distance)
                data.batch = torch.zeros((data.x.shape[0])).long()   # needed if mgn_hetero is used
                data.ptr = torch.tensor([0, data.batch.shape[0]])  # needed if mgn_hetero is used
                data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                data.to(device)

                old_position = predicted_position.to(device)
                velocity, h_0, c_0 = GNN(data, h_0, c_0)
                velocity = velocity.squeeze()
                predicted_position = old_position + velocity
                loss = F.mse_loss(predicted_position, data.y)
                time_difference_loss = F.mse_loss(old_position_true.to(device), data.y)
                if calculate_iou:
                    mesh_iou = calculate_mesh_iou(data.y.cpu().numpy(), predicted_position.detach().cpu().numpy(), triangles_grid, triangles_grid)
                else:
                    mesh_iou = 1

                # Losses
                rollout_predicted.append(predicted_position.detach().cpu().numpy())
                rollout_target.append(data.y.cpu().numpy())
                rollout_collider.append(collider_position.cpu().numpy())
                rollout_loss.append(loss.item()/loss_normalizer)
                rollout_iou.append(mesh_iou)
                rollout_time_difference_loss.append(time_difference_loss.item()/loss_normalizer)

                # Velocities
                mesh_velocity = (data.y - old_position_true.to(device))
                mean_mesh_x_velocity.append(torch.mean(mesh_velocity[:, 0]).item())
                mean_mesh_x_velocity_predicted.append(torch.mean(velocity[:, 0]).cpu())
                mean_mesh_y_velocity.append(torch.mean(mesh_velocity[:, 1]).item())
                mean_mesh_y_velocity_predicted.append(torch.mean(velocity[:, 1]).cpu())
                if mesh_velocity.shape[1] > 2:
                    mean_mesh_z_velocity.append(torch.mean(mesh_velocity[:, 2]).item())
                    mean_mesh_z_velocity_predicted.append(torch.mean(velocity[:, 2]).cpu())

        eval_loss.append(sum(rollout_loss)/len(rollout_loss))
        eval_iou.append(sum(rollout_iou)/len(rollout_iou))
        if detailed_print:
            print("rollout step eval loss in trajectory " + str(i).zfill(3) + ": ", sum(rollout_loss)/len(rollout_loss))
            print("average IOU in trajectory " + str(i).zfill(3) + ": ", sum(rollout_iou)/len(rollout_iou))
            print("time difference loss in trajectory " + str(i).zfill(3) + ": ", sum(rollout_time_difference_loss)/len(rollout_time_difference_loss))

        if show_curves:
            plot_velocity_curve(mean_mesh_x_velocity_predicted, mean_mesh_x_velocity, "Velocity x", save_animation=save_animation)
            plot_velocity_curve(mean_mesh_y_velocity_predicted, mean_mesh_y_velocity, "Velocity y", save_animation=save_animation)
            plot_loss_curve(rollout_loss, rollout_time_difference_loss, save_animation=save_animation)
            if rollout_predicted[0].shape[1] > 2:
                plot_velocity_curve(mean_mesh_z_velocity_predicted, mean_mesh_z_velocity, "Velocity z", save_animation=save_animation)
        if show_rollout:

            if rollout_predicted[0].shape[1] > 2:
                if "tube" in dataset:
                    animate_rollout3d_tube_mesh_overlay(ROOT_DIR, rollout_predicted, rollout_target, rollout_collider, triangles_grid,
                                               triangles_collider, fig_number=i,
                                               loss=sum(rollout_loss) / len(rollout_loss), stride=stride,
                                               save_animation=save_animation)
                else:
                    animate_rollout3d_mesh_side_by_side(ROOT_DIR, rollout_predicted, rollout_target, rollout_collider, triangles_grid,
                                               triangles_collider, fig_number=i,
                                               loss=sum(rollout_loss) / len(rollout_loss), stride=stride,
                                               save_animation=save_animation)
            else:

                animate_rollout_mesh_overlay(ROOT_DIR, rollout_predicted, rollout_target, rollout_collider, triangles_grid,
                                             triangles_collider, fig_number=i,
                                             loss=sum(rollout_loss) / len(rollout_loss), stride=stride,
                                             save_animation=save_animation)

    if detailed_print:
        print("\n ############################# SUMMARY ################################ \n")

        print("average rollout eval loss over all trajectories: ", np.mean(eval_loss))
        print("median rollout eval loss over all trajectories: ", np.median(eval_loss))
        print("average IOU over all trajectories: ", np.mean(eval_iou))
        print("Best Rollout Number: ", np.argmin(eval_loss), ", loss: ", min(eval_loss))
        print("Worst Rollout Number: ", np.argmax(eval_loss), ", loss: ", max(eval_loss))
        print("Loss standard deviation: ", np.std(eval_loss))
    else:
         print("MSE", np.mean(eval_loss))
         print("IOU", np.mean(eval_iou))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=str, default='example_models/plate/lstm', help="the name of the run in the 'data' directory")
    parser.add_argument("-s", "--split", type=str, default="test", help="The split on which to test on")
    parser.add_argument("-n", "--n_step", type=int, default=400, help="How many steps to perform rollout, without new gt mesh")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Which device to use")
    parser.add_argument("--show_curves", default=False, action=argparse.BooleanOptionalAction, help="Enable to show loss curves")
    parser.add_argument("--save_animation", default=False, action=argparse.BooleanOptionalAction, help="Enable to save animation")
    parser.add_argument("--show_rollout", default=True, action=argparse.BooleanOptionalAction, help="Enable to show rollout visualization")
    parser.add_argument("--mode", default=0, help="Evaluation mode, e.g. k_hop for using a point cloud only every k-th time step")
    parser.add_argument("--mode_k", type=int, default=0, help="k for evaluation, e.g. for k_hop it is the point cloud frequency")
    parser.add_argument("--start", type=int, default=0, help="Start trajectory")
    parser.add_argument("--stop", type=int, default=400, help="Stop trajectory")
    parser.add_argument("--crop", type=int, default=0, help="Steps to crop trajectory at the start")
    parser.add_argument("--loss_normalizer", type=float, default=1e-6, help="Normalizer for the loss")
    parser.add_argument('--model_name', type=str, default=None, help="Specify the name of the model to load parameters from")
    parser.add_argument('--dataset', type=str, default=None, help="Specify the name of a different dataset in the run directory you want to test on")
    parser.add_argument("--stride", type=int, default=1, help="Stride when showing rollout")
    parser.add_argument("--iou", default=False, action=argparse.BooleanOptionalAction, help="Additionally calculate the IoU (only 2d data)")
    parser.add_argument("--print", default=True, action=argparse.BooleanOptionalAction, help="Print detailed results for all trajectories")
    args = parser.parse_args()

    evaluate(args)
