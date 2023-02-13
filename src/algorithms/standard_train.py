import copy
import torch
import os
from torch_geometric.loader import DataLoader
import torch.nn as nn
from tqdm import tqdm
import wandb
from typing import Dict

from src.utils.eval_utils import single_step_evaluation, n_step_evaluation, test_model, n_plus_m_step_evaluation, mpc_m_mesh_evaluation
from src.utils.train_utils import get_network_config, log_gradients, seed_worker, seed_all, \
    add_noise_to_mesh_nodes, \
    add_pointcloud_dropout, calculate_loss_normalizer, add_noise_to_pcd_points
from src.utils.data_utils import convert_to_hetero_data, get_feature_info_from_data, get_shortest_trajectory, \
    count_parameters, convert_trajectory_to_data_list, crop_list_of_trajectories, predict_velocity, transform_position_to_edges
from modules.gnn_modules.get_gnn_model import get_gnn_model
from src.utils.wandb_utils import wandb_init, wandb_loss_logger, wandb_config_update
from src.utils.dataset_utils import build_dataset_for_split
from src.utils.get_connectivity_setting import get_connectivity_setting
from modules.datasets.Datasets import WeightedDataset


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
    input_pcd_noise = config.get("input_pcd_noise")

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
        trajectory_list_train = crop_list_of_trajectories(trajectory_list_train, 0, sequence_length)
        train_data_list = convert_trajectory_to_data_list(trajectory_list_train + trajectory_list_train_mgn)
        train_dataset = WeightedDataset(train_data_list, len_trajectory, len(trajectory_list_train), sequence_length, pcd_weighting)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    else:
        train_data_list = convert_trajectory_to_data_list(trajectory_list_train)
        trainloader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker)

    # prepare eval data
    eval_data_list = convert_trajectory_to_data_list(trajectory_list_eval)
    evalloader_single = DataLoader(eval_data_list, batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker)

    if loss_normalizer == 'None' or loss_normalizer is None:
        loss_normalizer = calculate_loss_normalizer(trainloader, device)

    # initialize best losses
    best_train_loss = float("inf")
    best_eval_loss_single_step = float("inf")
    best_eval_loss_10_step = float("inf")
    best_eval_loss_complete_rollout = float("inf")
    if additional_eval:
        eval_mode_list = ['k-hop', 'k-hop', 'k-hop', 'mpc10']
        eval_mode_name_list = ['eval loss complete rollout/2-hop',
                                'eval loss complete rollout/5-hop',
                               'eval loss complete rollout/10-hop',
                               'eval loss imputation mpc/10-mesh']
        k_list = [2, 5, 10, 0]
        m_list = [0, 0, 0, 10]
        additional_eval_list = [float("inf"), float("inf"), float("inf"), float("inf"), float("inf")]

    # main training loop
    optimizer = torch.optim.Adam(GNN.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    for epoch in range(1, num_epochs + 1):
        GNN.train()
        pbar = tqdm(total=len(trainloader))
        pbar.set_description(f'Training epoch: {epoch:04d}')
        total_loss = 0
        for data in trainloader:
            optimizer.zero_grad()
            data = add_pointcloud_dropout(data, pointcloud_dropout, hetero, use_world_edges)
            data.to(device)
            data = add_noise_to_mesh_nodes(data, input_mesh_noise, device)
            data = add_noise_to_pcd_points(data, input_pcd_noise, device)
            data = transform_position_to_edges(data, euclidian_distance)
            data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
            node_features_out, _, _ = GNN(data)
            velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
            predicted_position = data.y_old + velocity
            loss = criterion(predicted_position, data.y)
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
            eval_loss_single_step = single_step_evaluation(GNN, hetero, evalloader_single, device, epoch, loss_normalizer, euclidian_distance, use_color, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)
            eval_loss_10_step = n_step_evaluation(GNN, hetero, 10, trajectory_list_eval_raw, device, epoch, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)
            eval_loss_complete_rollout = n_step_evaluation(GNN, hetero, shortest_eval_trajectory, trajectory_list_eval_raw, device, epoch, loss_normalizer, edge_radius_dict, input_timestep, 0, 0, euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)

            # log evaluation losses using wandb
            best_eval_loss_single_step = wandb_loss_logger(wandb_log, eval_loss_single_step, "eval loss/single step", epoch, best_eval_loss_single_step, GNN, run_folder_dir)
            best_eval_loss_10_step = wandb_loss_logger(wandb_log, eval_loss_10_step, "eval loss/10 step", epoch, best_eval_loss_10_step, GNN, run_folder_dir)
            best_eval_loss_complete_rollout = wandb_loss_logger(wandb_log, eval_loss_complete_rollout, "eval loss/complete rollout", epoch, best_eval_loss_complete_rollout, GNN, run_folder_dir)

            if additional_eval:
                for i, mode in enumerate(eval_mode_list):
                    m = m_list[i]
                    if m == 0:
                        current_eval_loss_complete_rollout = n_step_evaluation(GNN, hetero, shortest_eval_trajectory, trajectory_list_eval_raw, device, epoch, loss_normalizer, edge_radius_dict, input_timestep, mode, k_list[i], euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)
                        additional_eval_list[i] = wandb_loss_logger(wandb_log, current_eval_loss_complete_rollout, eval_mode_name_list[i], epoch, additional_eval_list[i], GNN, run_folder_dir)
                    else:
                        if tissue_task:
                            mpc_hop = 20
                        else:
                            mpc_hop = 10
                        current_eval_loss = mpc_m_mesh_evaluation(GNN, hetero, m, mpc_hop, trajectory_list_eval_raw, device, epoch, loss_normalizer, edge_radius_dict, input_timestep, 'full-pcd', 'mesh-only', euclidian_distance, use_color, use_poisson, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)
                        additional_eval_list[i] = wandb_loss_logger(wandb_log, current_eval_loss, eval_mode_name_list[i], epoch, additional_eval_list[i], GNN, run_folder_dir)

    if wandb_log:
        if final_test:
            test_model(GNN, input_dataset, ROOT_DIR, run_folder_dir, "test", edge_radius_dict, input_timestep, euclidian_distance, hetero, use_color, use_poisson, tissue_task, batch_size, device, loss_normalizer, additional_eval, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)
            if "continuous_materials" in input_dataset:
                test_model(GNN, input_dataset, ROOT_DIR, run_folder_dir, "discrete_test", edge_radius_dict, input_timestep, euclidian_distance, hetero, use_color, use_poisson, tissue_task, batch_size, device, loss_normalizer, additional_eval, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)
                test_model(GNN, input_dataset, ROOT_DIR, run_folder_dir, "extrapolate_test", edge_radius_dict, input_timestep, euclidian_distance, hetero, use_color, use_poisson, tissue_task, batch_size, device, loss_normalizer, additional_eval, use_world_edges, use_mesh_coordinates, mgn_hetero, input_pcd_noise)


    wandb.finish()
