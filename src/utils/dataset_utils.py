import torch
import pickle
from tqdm import tqdm
import numpy as np
import os

from util.Types import *
import src.utils.graph_utils as graph_utils


def get_shortest_trajectory_dict(trajectory_dict_list: list, key: str) -> int:
    """
    Calculates the length of the shortest trajectory in a list of trajectory dicts for the given key
    Args:
        trajectory_dict_list: List of trajectory dicts
        key: key for the dictionary
    Returns:
        minimum: Minimum length of trajectory
    """
    minimum = len(trajectory_dict_list[0][key])
    for dictionary in trajectory_dict_list:
        trajectory = dictionary[key]
        min_i = len(trajectory)
        if min_i < minimum:
            minimum = min_i
    return minimum


def get_mesh_triangles_from_sofa(input_dataset: str, path: str, split: str, output_list: bool = False, tissue_task: bool = False):
    """
    Extract the mesh triangles from the data from SOFA
    Needed e.g. for plotting
    Args:
        input_dataset: Name of input dataset
        path: working directory
        split: name of split (train, eval, test)
        output_list: True if the list of triangles should be output
        tissue_task: True if 3D data is used

    Returns: Tuple
        triangles_grid: Mesh triangles for the deformable object
        triangles_collider: Mesh triangles for the collider
    """
    with open(os.path.join(path, "data/sofa", input_dataset, input_dataset + "_" + split + ".pkl"), "rb") as file:
        rollout_data = pickle.load(file)
    if output_list:
        triangles_grid = []
        triangles_collider = []
        for mesh_rollout_dict in rollout_data:
            if tissue_task:
                collider = np.empty(1)
                grid = mesh_rollout_dict["tissue_mesh_triangles"]
            else:
                collider = mesh_rollout_dict["triangles_collider"]
                grid = mesh_rollout_dict["triangles_grid"]
            triangles_grid.append(grid)
            triangles_collider.append(collider)
    else:
        mesh_rollout_dict = rollout_data[0]
        if tissue_task:
            triangles_collider = np.empty(1)
            triangles_grid = mesh_rollout_dict["tissue_mesh_triangles"]
        else:
            triangles_collider = mesh_rollout_dict["triangles_collider"]
            triangles_grid = mesh_rollout_dict["triangles_grid"]

    return triangles_grid, triangles_collider


def get_datapoint_from_sofa(input_dataset: str, path: str, split: str, traj_index: int, timestep: int):
    """
    Extract a single data point from SOFA
    Args:
        input_dataset: Name of input dataset
        path: working directory
        split: name of split (train, eval, test)
        traj_index: trajectory index
        timestep: time step index of datapoint

    Returns:
    """
    with open(os.path.join(path, "data/sofa", input_dataset, input_dataset + "_" + split + ".pkl"), "rb") as file:
        rollout_data = pickle.load(file)

    data_dict = rollout_data[traj_index]
    if 'tube' in input_dataset:
        time_dict = {"tube_mesh_positions": data_dict["tissue_mesh_positions"][timestep],
                    "tube_mesh_triangles": data_dict["tissue_mesh_triangles"],
                    "gripper_mesh_positions": data_dict["gripper_position"][timestep],
                    "gripper_mesh_triangles": data_dict["gripper_triangles"],
                    "panda_mesh_positions": data_dict["panda_position"][timestep],
                    "panda_mesh_triangles": data_dict["panda_triangles"],
                    "point_cloud_positions": data_dict["tissue_pcd_points"][timestep]}
    elif 'tissue' in input_dataset:
        time_dict = {"tissue_mesh_positions": data_dict["tissue_mesh_positions"][timestep],
                    "tissue_mesh_triangles": data_dict["tissue_mesh_triangles"],
                    "gripper_mesh_positions": data_dict["gripper_mesh_positions"][timestep],
                    "gripper_mesh_triangles": data_dict["gripper_mesh_triangles"],
                    "liver_mesh_positions": data_dict["liver_mesh_positions"],
                    "liver_mesh_triangles": data_dict["liver_mesh_triangles"],
                    "point_cloud_positions": data_dict["tissue_pcd_points"][timestep]}


    else:
        raise NotImplementedError
    return time_dict


def get_trajectory_from_sofa(input_dataset: str, path: str, split: str, traj_index: int):
    """
    Extract one trajecotry from SOFA data
    Args:
        input_dataset: Name of input dataset
        path: working directory
        split: name of split (train, eval, test)
        traj_index: index of trajectory

    Returns:
    """
    with open(os.path.join(path, "data/sofa", input_dataset, input_dataset + "_" + split + ".pkl"), "rb") as file:
        rollout_data = pickle.load(file)

    data_dict = rollout_data[traj_index]
    if 'tube' in input_dataset:
        time_dict = {"tube_mesh_positions": data_dict["tissue_mesh_positions"],
                    "tube_mesh_triangles": data_dict["tissue_mesh_triangles"],
                    "gripper_mesh_positions": data_dict["gripper_position"],
                    "gripper_mesh_triangles": data_dict["gripper_triangles"],
                    "panda_mesh_positions": data_dict["panda_position"],
                    "panda_mesh_triangles": data_dict["panda_triangles"],
                    "point_cloud_positions": data_dict["tissue_pcd_points"]}
    elif 'tissue' in input_dataset:
        time_dict = {"tissue_mesh_positions": data_dict["tissue_mesh_positions"],
                    "tissue_mesh_triangles": data_dict["tissue_mesh_triangles"],
                    "gripper_mesh_positions": data_dict["gripper_mesh_positions"],
                    "gripper_mesh_triangles": data_dict["gripper_mesh_triangles"],
                    "liver_mesh_positions": data_dict["liver_mesh_positions"],
                    "liver_mesh_triangles": data_dict["liver_mesh_triangles"],
                    "point_cloud_positions": data_dict["tissue_pcd_points"]}


    else:
        raise NotImplementedError
    return time_dict



def prepare_data_for_trajectory(data: Tuple, timestep: int, input_timestep: str = 't+1', use_color: bool = False, tissue_task: bool = False) -> Tuple:
    """
    Function to get the correct data and convert to tensors from a single time step of a trajectory of the prepared data output from SOFA
    Args:
        data: Tuple of the data from the prepare_from_sofa function
        timestep: timestep in trajectory
        input_timestep: Use collider and point cloud from time step 't' or 't+1'
        use_color:
        tissue_task:

    Returns: Tuple
        grid_positions: Tensor containing point cloud positions
        collider_positions: Tensor containing collider positions
        mesh_positions: Tensor containing mesh positions
        input_mesh_edge_index: Tensor containing mesh edge indices
        label: Tensor containing mesh positions of the next time step
        grid_colors: Tensor containing colors for point cloud
        initial_mesh_positions: Tensor containing the initial mesh positions for mesh-edge generation (see MGN)
        next_collider_positions (only 3D): Tensor containing the position of the collider in the next time step (to calculate velocity)
        poisson (only 2D): Tensor containing poisson ratio of the current data sample
    """
    if tissue_task:
        pcd_positions_grid, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, poisson_ratio = data
    else:
        pcd_positions_grid, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, poisson_ratio = data
    if poisson_ratio is not None:
        poisson_ratio = torch.tensor(poisson_ratio)

    if input_timestep == 't+1':
        grid_positions = torch.tensor(pcd_positions_grid[timestep + 1])
        collider_positions = torch.tensor(nodes_collider[timestep + 1])
        mesh_positions = torch.tensor(nodes_grid[timestep])
        mesh_edge_index = torch.tensor(edge_index_grid.T).long()
        label = torch.tensor(nodes_grid[timestep + 1])
        next_collider_positions = torch.tensor(nodes_collider[timestep + 2])
        initial_mesh_positions = torch.tensor(nodes_grid[0])
        if use_color:
            grid_colors = torch.tensor(pcd_colors[timestep + 1])
        else:
            grid_colors = None
    elif input_timestep == 't':
        grid_positions = torch.tensor(pcd_positions_grid[timestep])
        collider_positions = torch.tensor(nodes_collider[timestep])
        mesh_positions = torch.tensor(nodes_grid[timestep])
        mesh_edge_index = torch.tensor(edge_index_grid.T).long()
        label = torch.tensor(nodes_grid[timestep + 1])
        next_collider_positions = torch.tensor(nodes_collider[timestep + 1])
        initial_mesh_positions = torch.tensor(nodes_grid[0])
        if use_color:
            grid_colors = torch.tensor(pcd_colors[timestep])
        else:
            grid_colors = None
    else:
        raise ValueError("input_timestep can only be t or t+1")

    if tissue_task:
        data = grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions.float(), next_collider_positions, poisson_ratio
    else:
        data = grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions.float(), poisson_ratio

    return data


#################################### dataset building ######################################

def build_dataset_for_split(input_dataset: str,
                            path: str,
                            split: str,
                            edge_radius_dict: Dict,
                            device,
                            input_timestep: str,
                            use_mesh_coordinates: bool,
                            hetero: bool = False,
                            raw: bool = False,
                            use_color: bool = False,
                            use_poisson: bool = False,
                            tissue_task: bool = False):
    """
    Choose correct dataset build function for 2D or 3D data
    """
    if tissue_task:
        return build_3d_dataset_for_split(input_dataset,
                                          path,
                                          split,
                                          edge_radius_dict,
                                          device,
                                          input_timestep,
                                          use_mesh_coordinates,
                                          hetero,
                                          raw,
                                          use_color,
                                          use_poisson)
    else:
        return build_2d_dataset_for_split(input_dataset,
                                          path,
                                          split,
                                          edge_radius_dict,
                                          device,
                                          input_timestep,
                                          use_mesh_coordinates,
                                          hetero,
                                          raw,
                                          use_color,
                                          use_poisson)






######################################### 2D Dataset #########################################

def prepare_data_from_sofa(mesh_rollout_dict: Dict, use_color: bool = False, use_poisson: bool = False) -> Tuple:
    """
    Prepares the dict from SOFA for 2D data, normalizes it and outputs a tuple with important data
    Args:
        mesh_rollout_dict: Dict from SOFA
        use_color: True if color textured gradient is used
        use_poisson: True if the Poisson's ratio is used

    Returns: Tuple of data for point cloud, collider and mesh
    """
    nodes_grid = mesh_rollout_dict["nodes_grid"]
    edge_index_grid = mesh_rollout_dict["edge_index_grid"]
    nodes_collider = mesh_rollout_dict["nodes_collider"]
    pcd_positions = mesh_rollout_dict["pcd_points"]
    pcd_colors = mesh_rollout_dict["pcd_colors"]
    if use_poisson:
        poisson_ratio = mesh_rollout_dict["poisson_ratio"]
        poisson_ratio = (poisson_ratio+0.205)*(200/139)  # normalize to -1,1
    else:
        poisson_ratio = None

    if not use_color:
        pcd_colors = None

    data = (pcd_positions, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, poisson_ratio)
    return data


def create_raw_graph(trajectory: Tuple, timestep: int, use_color: bool = False) -> Data:
    """
    Creates a raw graph which includes the basic entities for creating the common graph: point cloud, collider and mesh.
    The output is used e.g. in the n-step rollout to faster build a graph from the raw dataset
    Args:
        trajectory: Data of the current trajectory
        timestep: current time step in trajectory
        use_color:

    Returns:
        data: PyG Data element containing the raw graph
    """
    # get trajectory data for future time step
    grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions, poisson_ratio = prepare_data_for_trajectory(trajectory,
                                                                                                                                                                 timestep,
                                                                                                                                                                 input_timestep='t+1',
                                                                                                                                                                 use_color=use_color)
    pos1 = grid_positions
    pos2 = collider_positions
    pos3 = mesh_positions
    if grid_colors is not None:
        color_copy = grid_colors.float()
    else:
        color_copy = None
    if poisson_ratio is not None:
        poisson_ratio_copy = poisson_ratio
    else:
        poisson_ratio_copy = None

    # get trajectory data for current time step
    grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions, poisson_ratio = prepare_data_for_trajectory(trajectory,
                                                                                                                                                                 timestep,
                                                                                                                                                                 input_timestep='t',
                                                                                                                                                                 use_color=use_color)
    pos1_old = grid_positions
    pos2_old = collider_positions
    if grid_colors is not None:
        color_old = grid_colors.float()
    else:
        color_old = None
    if poisson_ratio is not None:
        poisson_ratio_old = poisson_ratio
    else:
        poisson_ratio_old = None

    # create PyG data object containing data for both time steps
    data = Data(grid_positions=pos1.float(),
                collider_positions=pos2.float(),
                mesh_positions=pos3.float(),
                grid_positions_old=pos1_old.float(),
                collider_positions_old=pos2_old.float(),
                mesh_edge_index=mesh_edge_index.long(),
                y=label.float(),
                y_old=mesh_positions.float(),
                grid_colors=color_copy,
                grid_colors_old=color_old,
                poisson_ratio=poisson_ratio_copy,
                poisson_ratio_old=poisson_ratio_old,
                initial_mesh_positions=initial_mesh_positions.float())

    return data


def build_2d_dataset_for_split(input_dataset: str,
                               path: str,
                               split: str,
                               edge_radius_dict: Dict,
                               device,
                               input_timestep: str,
                               use_mesh_coordinates: bool,
                               hetero: bool = False,
                               raw: bool = False,
                               use_color: bool = False,
                               use_poisson: bool = False) -> list:
    """
    Builds the trajectory list for the 2D input data
    Args:
        input_dataset: Name of input dataset
        path: working directory
        split: name of split (train, eval, test)
        edge_radius_dict: Dictionary containing the edge radii
        device: Working device, either cpu or cuda
        input_timestep: Use collider and point cloud from time step 't' or 't+1'
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        hetero: Does nothing if False
        raw: True to create a dataset without any edges created by nearst neighbor
        use_color: Color gradient texture for point cloud
        use_poisson: Poisson's ratio as feature
    Returns:
        trajectory_list: List of trajectories containing the data elements for each time step

    """

    # Load sofa data
    print(f"Generating {split} data")
    with open(os.path.join(path, "data/sofa", input_dataset, input_dataset + "_" + split + ".pkl"), "rb") as file:
        rollout_data = pickle.load(file)
    trajectory_list = []

    for index, trajectory in enumerate(tqdm(rollout_data)):
        rollout_length = len(trajectory["nodes_grid"])
        data_list = []
        trajectory = prepare_data_from_sofa(trajectory, use_color, use_poisson)

        for timestep in (range(rollout_length-2)):
            if raw:
                data = create_raw_graph(trajectory, timestep, use_color=use_color)
            else:

                # get trajectory data for current timestep
                data_timestep = prepare_data_for_trajectory(trajectory, timestep, input_timestep=input_timestep, use_color=use_color)

                # create nearest neighbor graph with the given radius dict
                if hetero:
                    data = graph_utils.create_hetero_graph_from_raw(data_timestep, edge_radius_dict=edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates)
                else:
                    data = graph_utils.create_graph_from_raw(data_timestep, edge_radius_dict=edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates)

            data_list.append(data)  # append object for timestep t to data_list
        trajectory_list.append(data_list)  # create list of trajectories with each trajectory being a list itself

    return trajectory_list






######################################### 3D Dataset #########################################

def prepare_3d_data_from_sofa(mesh_rollout_dict: Dict, use_color: bool = False, use_poisson: bool = False):
    """
    Prepares the dict from SOFA for 3D data, normalizes it and outputs a tuple with important data
    Args:
        mesh_rollout_dict: Dict from SOFA
        use_color: Not used
        use_poisson: Not used

    Returns: Tuple of data for point cloud, collider and mesh
    """
    nodes_grid = mesh_rollout_dict["tissue_mesh_positions"]
    edge_index_grid = mesh_rollout_dict["tissue_mesh_edges"]
    nodes_collider = mesh_rollout_dict["gripper_position"]
    pcd_positions = mesh_rollout_dict["tissue_pcd_points"]
    pcd_colors = None

    if not use_color:
        pcd_colors = None

    if use_poisson:
        poisson_ratio = mesh_rollout_dict["poisson_ratio"]
        poisson_ratio = (poisson_ratio+0.205)*(200/139)  # normalize to -1,1
    else:
        poisson_ratio = None

    data = (pcd_positions, nodes_collider, nodes_grid, edge_index_grid, pcd_colors, poisson_ratio)

    return data


def create_3d_raw_graph(trajectory: Tuple, timestep: int, use_color: bool = False) -> Data:
    """
    Creates a raw graph which includes the basic entities for creating the common graph: point cloud, gripper and mesh.
    The output is used e.g. in the n-step rollout to faster build a graph from the raw dataset
    Args:
        trajectory: Data of the current trajectory
        timestep: current time step in trajectory
        use_color:

    Returns:
        data: PyG Data element containing the raw graph
    """
    # get trajectory data for future time step
    grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions, next_collider_positions, poisson_ratio = prepare_data_for_trajectory(trajectory,
                                                                                                                                                                           timestep,
                                                                                                                                                                           input_timestep='t+1',
                                                                                                                                                                           use_color=use_color,
                                                                                                                                                                           tissue_task=True)
    pos1 = grid_positions
    pos2 = collider_positions
    pos3 = mesh_positions
    pos4 = next_collider_positions
    if grid_colors is not None:
        color1 = grid_colors.float()
    else:
        color1 = None
    if poisson_ratio is not None:
        poisson_ratio_copy = poisson_ratio
    else:
        poisson_ratio_copy = None

    # get trajectory data for current time step
    grid_positions, collider_positions, mesh_positions, mesh_edge_index, label, grid_colors, initial_mesh_positions, next_collider_positions, poisson_ratio = prepare_data_for_trajectory(trajectory,
                                                                                                                                                                           timestep,
                                                                                                                                                                           input_timestep='t',
                                                                                                                                                                           use_color=use_color,
                                                                                                                                                                           tissue_task=True)
    pos1_old = grid_positions
    pos2_old = collider_positions
    pos3_old = next_collider_positions
    if grid_colors is not None:
        color1_old = grid_colors.float()
    else:
        color1_old = None
    if poisson_ratio is not None:
        poisson_ratio_old = poisson_ratio
    else:
        poisson_ratio_old = None

    # create data object for torch
    data = Data(grid_positions=pos1.float(),
                collider_positions=pos2.float(),
                next_collider_positions=pos4.float(),
                mesh_positions=pos3.float(),
                grid_positions_old=pos1_old.float(),
                collider_positions_old=pos2_old.float(),
                next_collider_positions_old=pos3_old.float(),
                mesh_edge_index=mesh_edge_index.long(),
                y=label.float(),
                y_old=mesh_positions.float(),
                grid_colors=color1,
                grid_colors_old=color1_old,
                initial_mesh_positions=initial_mesh_positions.float(),
                poisson_ratio=poisson_ratio_copy,
                poisson_ratio_old=poisson_ratio_old)

    return data


def build_3d_dataset_for_split(input_dataset: str,
                               path: str,
                               split: str,
                               edge_radius_dict: Dict,
                               device,
                               input_timestep: str,
                               use_mesh_coordinates: bool = False,
                               hetero: bool = False,
                               raw: bool = False,
                               use_color: bool = False,
                               use_poisson: bool = False) -> object:
    """
    Builds the trajectory list for the 3D input data
    Args:
        input_dataset: Name of input dataset
        path: working directory
        split: name of split (train, eval, test)
        edge_radius_dict: Dictionary containing the edge radii
        device: Working device, either cpu or cuda
        input_timestep: Use collider and point cloud from time step 't' or 't+1'
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        hetero: Does nothing if False
        raw: True to create a dataset without any edges created by nearst neighbor
        use_color: Color gradient texture for point cloud
        use_poisson:
    Returns:
        trajectory_list: List of trajectories containing the data elements for each time step

    """
    # Load sofa data
    print(f"Generating {split} data")
    with open(os.path.join(path, "data/sofa", input_dataset, input_dataset + "_" + split + ".pkl"), "rb") as file:
        rollout_data = pickle.load(file)

    # Limit all trajectories to the same length
    min_rollout_length = get_shortest_trajectory_dict(rollout_data, "tissue_mesh_positions")
    if min_rollout_length % 5 == 0:
        rollout_length = min_rollout_length
    else:
        rollout_length = min_rollout_length - min_rollout_length % 5
    trajectory_list = []

    for index, trajectory in enumerate(tqdm(rollout_data)):
        if split == 'train':
            rollout_length = len(trajectory["tissue_mesh_positions"])
        data_list = []
        trajectory = prepare_3d_data_from_sofa(trajectory, use_color, use_poisson)

        for timestep in (range(0, rollout_length-5)):
            if raw:
                data = create_3d_raw_graph(trajectory, timestep, use_color=use_color)
            else:
                #get trajectory data for current timestep
                data_timestep = prepare_data_for_trajectory(trajectory, timestep, input_timestep=input_timestep, use_color=use_color, tissue_task=True)
                # create nearest neighbor graph with the given radius
                if hetero:
                    data = graph_utils.create_hetero_graph_from_raw(data_timestep, edge_radius_dict=edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates, tissue_task=True)
                else:
                    data = graph_utils.create_graph_from_raw(data_timestep, edge_radius_dict=edge_radius_dict, output_device=device, use_mesh_coordinates=use_mesh_coordinates, tissue_task=True)

            data_list.append(data)  # append object for timestep t to data_list
        trajectory_list.append(data_list)  # create list of trajectories with each trajectory being a list itself

    return trajectory_list
