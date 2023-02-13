import torch
import numpy as np
import copy
import yaml
from torch.nn import functional as F
import os
import argparse

from src.utils.eval_utils import get_radius_dict_for_evaluation_mode, get_save_dict_value, calculate_mesh_iou, calculate_mesh_iou_queue
from src.utils.graph_utils import create_graph_from_raw
from src.utils.mesh_utils import generate_mesh_from_pcd_hull
from src.utils.dataset_utils import build_dataset_for_split, get_mesh_triangles_from_sofa, get_trajectory_from_sofa
from src.utils.get_connectivity_setting import get_connectivity_setting
from src.utils.data_utils import get_timestep_data, convert_to_hetero_data, get_feature_info_from_data, \
    count_parameters, predict_velocity, transform_position_to_edges
from modules.gnn_modules.get_gnn_model import get_gnn_model
from src.utils.visualization.visualization import plot_loss_curve, plot_velocity_curve
from src.utils.visualization.visualization2d import animate_rollout_mesh_overlay, animate_rollout_pcd_mesh_overlay
from src.utils.visualization.visualization3d import animate_rollout3d_mesh_overlay, animate_rollout3d_tube_mesh_overlay


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
    run_name2 = args.run2
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
    initial_pcd2mesh = args.initial_pcd2mesh
    pcd2mesh = args.pcd2mesh
    meshing_method = args.meshing

    if initial_pcd2mesh or pcd2mesh:
        import pymesh
        import multiprocessing

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
                # todo
                #dataset = config.get("connectivity_setting")['value']
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
    if "tube" in dataset:
        traj_dict = get_trajectory_from_sofa(input_dataset, ROOT_DIR, eval_mode, 0)
        triangles_collider = traj_dict["gripper_mesh_triangles"]
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

    # perform 2 GNN rollout
    two_gnn = False
    if run_name2 is not None:
        two_gnn = True
        # build and load GNN
        GNN2 = get_gnn_model(in_node_features, in_edge_features, in_global_features, out_node_features, network_config, hetero, device)
        if detailed_print:
            count_parameters(GNN2, show_details=False)
        GNN2.load_state_dict(torch.load(os.path.join(ROOT_DIR, "models", run_name2, "GNN_model_" + str(epoch)), map_location=device))
        GNN2.eval()

    eval_loss = []
    eval_iou = []
    for i in range(len(trajectory_list_raw)):
        rollout_predicted = []
        rollout_predicted_mesh = []
        rollout_predicted_faces = []
        rollout_target = []
        rollout_collider = []
        rollout_pcd = []
        rollout_loss = []
        rollout_iou = []
        rollout_time_difference_loss = []
        mean_mesh_x_velocity = []
        mean_mesh_x_velocity_predicted = []
        mean_mesh_y_velocity = []
        mean_mesh_y_velocity_predicted = []
        mean_mesh_z_velocity = []
        mean_mesh_z_velocity_predicted = []

        with torch.no_grad():
            for index, data in enumerate(trajectory_list_raw[i][crop_steps:]):
                old_position_true = data.y_old.float()
                collider_position = data.collider_positions
                pcd_positions = data.grid_positions
                if index % n_step == 0:
                    if initial_pcd2mesh:
                        # calculate mesh from pcd and replace it in the data object
                        hull_mesh = generate_mesh_from_pcd_hull(data, "t", meshing_method, detailed_print=detailed_print)
                        data.y_old = torch.tensor(hull_mesh.vertices).float()
                        data.mesh_positions = torch.tensor(hull_mesh.vertices).float()
                        triangles_grid_pred = torch.tensor(hull_mesh.faces)
                        _, hull_mesh_edges = pymesh.mesh_to_graph(hull_mesh)
                        predicted_position = data.y_old
                    else:
                        predicted_position = trajectory_list_raw[i][crop_steps+index].y_old

                if initial_pcd2mesh:
                    data.initial_mesh_positions = torch.tensor(hull_mesh.vertices).float()
                    data.mesh_edge_index = torch.tensor(hull_mesh_edges.transpose()).long()
                    current_hull_mesh = hull_mesh

                if pcd2mesh:
                    current_hull_mesh = generate_mesh_from_pcd_hull(data, "t+1", meshing_method, detailed_print=detailed_print)

                # predict mesh at t+1 by using grid and collider of step t or t+1
                data_timestep = get_timestep_data(data, predicted_position, input_timestep, use_color, use_poisson, tissue_task)
                current_edge_radius_dict = get_radius_dict_for_evaluation_mode(index, edge_radius_dict, mode, k)
                data = create_graph_from_raw(data_timestep, edge_radius_dict=current_edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates,
                                             hetero=hetero, tissue_task=tissue_task)
                data = transform_position_to_edges(data, euclidian_distance)
                data.batch = torch.zeros((data.x.shape[0])).long()   # needed if mgn_hetero is used
                data.ptr = torch.tensor([0, data.batch.shape[0]])  # needed if mgn_hetero is used
                data = convert_to_hetero_data(data, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
                data.to(device)

                # predict next mesh
                if current_edge_radius_dict == edge_radius_dict:
                    node_features_out, _, _ = GNN(data)
                else:
                    if two_gnn:
                        node_features_out, _, _ = GNN2(data)
                    else:
                        node_features_out, _, _ = GNN(data)
                old_position = predicted_position.to(device)
                velocity = predict_velocity(node_features_out, data, hetero, mgn_hetero)
                predicted_position = old_position + velocity

                # calculate loss
                if initial_pcd2mesh or pcd2mesh:
                    loss = float('nan')
                    time_difference_loss = float('nan')
                else:
                    loss = F.mse_loss(predicted_position, data.y).item()
                    time_difference_loss = F.mse_loss(old_position_true.to(device), data.y).item()
                if calculate_iou:
                    if pcd2mesh:
                        # calculate iou for pcd2mesh in each time step sometimes crashes, so we need the multiprocessing queue
                        queue = multiprocessing.Queue()
                        queue.put({'iou': None})
                        p = multiprocessing.Process(target=calculate_mesh_iou_queue, name="Foo", args=(data.y.cpu().numpy(), triangles_grid, current_hull_mesh, queue))
                        p.start()
                        p.join(2)
                        mesh_iou = queue.get()['iou']

                        # If thread is active
                        if p.is_alive():
                            print("killed one mesh_iou process")
                            p.terminate()
                            p.join()
                    elif initial_pcd2mesh:
                        mesh_iou = calculate_mesh_iou(data.y.cpu().numpy(), predicted_position.detach().cpu().numpy(), triangles_grid, triangles_grid_pred)
                    else:
                        mesh_iou = calculate_mesh_iou(data.y.cpu().numpy(), predicted_position.detach().cpu().numpy(), triangles_grid, triangles_grid)
                else:
                    mesh_iou = float('nan')

                # Losses
                rollout_predicted.append(predicted_position.detach().cpu().numpy())
                rollout_target.append(data.y.cpu().numpy())
                rollout_collider.append(collider_position.cpu().numpy())
                rollout_pcd.append(pcd_positions.cpu().numpy())
                rollout_loss.append(loss/loss_normalizer)
                if mesh_iou is not None:
                    rollout_iou.append(mesh_iou)
                rollout_time_difference_loss.append(time_difference_loss/loss_normalizer)
                if initial_pcd2mesh or pcd2mesh:
                    rollout_predicted_mesh.append(np.asarray(current_hull_mesh.vertices))
                    rollout_predicted_faces.append(np.asarray(current_hull_mesh.faces))

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
                    animate_rollout3d_mesh_overlay(ROOT_DIR, rollout_predicted, rollout_target, rollout_collider, triangles_grid,
                                               triangles_collider, fig_number=i,
                                               loss=sum(rollout_loss) / len(rollout_loss), stride=stride,
                                               save_animation=save_animation)
            else:
                if pcd2mesh:
                    animate_rollout_mesh_overlay(ROOT_DIR, rollout_predicted_mesh, rollout_target, rollout_collider, triangles_grid,
                                                 triangles_collider, rollout_predicted_faces, fig_number=i,
                                                 loss=10, stride=stride,
                                                 save_animation=save_animation)
                else:
                    if initial_pcd2mesh:
                        animate_rollout_mesh_overlay(ROOT_DIR, rollout_predicted, rollout_target, rollout_collider, triangles_grid,
                                                 triangles_collider, triangles_grid_pred, fig_number=i,
                                                 loss=100, stride=stride,
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
        if not(initial_pcd2mesh or pcd2mesh):
            print("Best Rollout Number: ", np.argmin(eval_loss), ", loss: ", min(eval_loss))
            print("Worst Rollout Number: ", np.argmax(eval_loss), ", loss: ", max(eval_loss))
            print("Loss standard deviation: ", np.std(eval_loss))
        else:
            print("Best Rollout Number: ", np.argmax(eval_iou), ", IOU: ", max(eval_iou))
            print("Worst Rollout Number: ", np.argmin(eval_iou), ", IOU: ", min(eval_iou))
    else:
        if calculate_iou:
            print("MSE", np.mean(eval_loss))
            print("IOU", np.mean(eval_iou))
        else:
            print(np.mean(eval_loss))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=str, default='example_models/plate/ggns', help="The name of the run in the 'data' directory")
    parser.add_argument("--run2", default=None, help="If two gnns are used, the name of the second run in the 'data' directory")
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
    parser.add_argument("--initial_pcd2mesh", default=False, action=argparse.BooleanOptionalAction, help="Construct initial mesh from pcd and then roll out")
    parser.add_argument("--pcd2mesh", default=False, action=argparse.BooleanOptionalAction, help="Construct mesh from pcd in all time steps")
    parser.add_argument("--meshing", type=str, default="alpha_shapes_sub", help="Method to create the mesh from point cloud")
    args = parser.parse_args()

    evaluate(args)
