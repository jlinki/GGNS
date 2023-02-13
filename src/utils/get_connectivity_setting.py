from util.Types import *


def get_connectivity_setting(dataset: str) -> Tuple:
    """
    Outputs the corresponding properties: edge radii, input time step and euclidean_distance for the used dataset
    Dataset here refers to the method how the graph is build from the input data which comes from SOFA
    Args:
        dataset: Name of that specifies the dataset
    Returns: Tuple.
        edge_radius_dict: Resulting edge radius dict for dataset
        input_timestep: Use point cloud of time step 't' or 't+1'
        euclidian_distance: Is a Euclidean distance as edges feature used
        tissue_task: Is this a 3D dataset
    """
    # 2D Deformable Plate connectivity settings
    if dataset == "coarse_meshgraphnet_t":
        edge_radius = [0.0, 0.08, None, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_meshgraphnet_world_t":
        edge_radius = [0.0, 0.08, 0.35, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]  # 0.25 smallest mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_full_graph_t":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_pcd_edges_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_pcd_edges_no_col_t":
        edge_radius = [0.0, 0.0, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.0, 0.08, 0.08, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_larger_radius_t":
        edge_radius = [0.0, 0.08, None, 0.18, 0.2, 0.0, 0.18, 0.2, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_equal_edges_t":
        edge_radius = [0.2, 0.08, None, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        input_timestep = "t"
        euclidian_distance = True

    # 3D Tisse Manipulation connectivity settings
    elif dataset == "tissue_meshgraphnet_t":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_world_t":
        edge_radius = [0.0, 0.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4]  # 0.068 # smalles mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_no_pcd_edges_t":
        edge_radius = [0.0, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_full_graph_t":
        edge_radius = [0.07, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_reduced_pcd_edges_t":
        edge_radius = [0.05, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True

    # 3D Cavity Grasping connectivity settings
    elif dataset == "cavity_meshgraphnet":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_meshgraphnet_world":
        edge_radius = [0.0, 0.0, 0.07, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_no_pcd_edges":
        edge_radius = [0.0, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_no_pcd_edges_nc":
        edge_radius = [0.0, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_full_graph":
        edge_radius = [0.05, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "cavity_no_collider":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        input_timestep = "t"
        euclidian_distance = True

    else:
        raise ValueError(f"Dataset {dataset} does currently not exist. Consider adding it to get_connectivity_setting")

    edge_radius_dict = get_radius_dict(edge_radius)

    if "tissue" in dataset:
        tissue_task = True
    elif "cavity" in dataset:
        tissue_task = True
    else:
        tissue_task = False

    return edge_radius_dict, input_timestep, euclidian_distance, tissue_task


def get_radius_dict(edge_radius: list) -> Dict:
    """
    Build an edge radius dict from a list of edge radii
    Args:
        edge_radius: List of the used edge radii
    Returns:
        edge_radius_dict: Dict containing the edge radii with their names
    """
    edge_radius_keys = [('grid', '0', 'grid'),
                        ('collider', '1', 'collider'),
                        ('mesh', '2', 'mesh'),
                        ('collider', '3', 'grid'),
                        ('mesh', '4', 'grid'),
                        ('mesh', '5', 'collider'),
                        ('grid', '6', 'collider'),
                        ('grid', '7', 'mesh'),
                        ('collider', '8', 'mesh')]

    edge_radius_dict = dict(zip(edge_radius_keys, edge_radius))
    return edge_radius_dict


############################## OLD CONNECTIVITIES ###############################

def get_old_dataset_properties(dataset: str) -> Tuple:
    """
    OLD VERSION
    Outputs the corresponding properties: edge radii, input time step and euclidean_distance for the used dataset
    Dataset here refers to the method how the graph is build from the input data which comes from SOFA
    Args:
        dataset: Name of that specifies the dataset
    Returns: Tuple.
        edge_radius_dict: Resulting edge radius dict for dataset
        input_timestep: Use point cloud of time step 't' or 't+1'
        euclidian_distance: Is a Euclidean distance as edges feature used
        tissue_task: Is this a 3D dataset
    """
    # 2D data
    if dataset == "coarse_meshgraphnet_t":
        edge_radius = [0.0, 0.08, None, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_meshgraphnet_world_t":
        edge_radius = [0.0, 0.08, 0.35, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]  # 0.25 smallest mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_full_graph_t":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_pcd_edges_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_pcd_edges_no_col_t":
        edge_radius = [0.0, 0.0, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.0, 0.08, 0.08, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_larger_radius_t":
        edge_radius = [0.0, 0.08, None, 0.18, 0.2, 0.0, 0.18, 0.2, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_equal_edges_t":
        edge_radius = [0.2, 0.08, None, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        input_timestep = "t"
        euclidian_distance = True

    # 3D Tisse Manipulation connectivity settings
    elif dataset == "tissue_meshgraphnet_t":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_world_t":
        edge_radius = [0.0, 0.0, 0.01, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02]  # 0.0034 smalles mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_no_pcd_edges_t":
        edge_radius = [0.0, 0.0, None, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_full_graph_t":
        edge_radius = [0.0035, 0.0, None, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_reduced_pcd_edges_t":
        edge_radius = [0.0025, 0.0, None, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    #################### old ########################
    elif dataset == "dataset_coarse+":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "dataset_coarse2+":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t+1"
        euclidian_distance = False
    elif dataset == "dataset_coarse3+":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "dataset_coarse4+":
        edge_radius = [0.1, 0.08, None, 0.16, 0.12, 0.3, 0.16, 0.12, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse5":
        edge_radius = [0.12, 0.08, None, 0.18, 0.18, 0.3, 0.18, 0.18, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "more_grid_edges":
        edge_radius = [0.2, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_mesh_only":
        edge_radius = [0.0, 0.00, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_longer_steps_t":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_longer_steps_t+1":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_meshgraphnet_t+1":
        edge_radius = [0.0, 0.08, None, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "more_edges":
        edge_radius = [0.12, 0.08, None, 0.18, 0.18, 0.3, 0.18, 0.18, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_no_grid":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_02":
        edge_radius = [0.0, 0.08, None, 0.08, 0.2, 0.3, 0.08, 0.2, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_03":
        edge_radius = [0.0, 0.08, None, 0.08, 0.3, 0.3, 0.08, 0.3, 0.3]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_02_all":
        edge_radius = [0.0, 0.08, None, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_mesh_grid-collider":
        edge_radius = [0.0, 0.08, None, 0.18, 0.2, 0.0, 0.18, 0.2, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_mesh_grid-collider2":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.0, 0.08, 0.08, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_no_grid-grid_mesh-grid_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.0, 0.08, 0.08, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_grid-grid_mesh-grid2_t":
        edge_radius = [0.0, 0.08, None, 0.18, 0.2, 0.0, 0.18, 0.2, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_grid-grid_mesh-grid3_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.2, 0.0, 0.08, 0.2, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_mesh_grid":
        edge_radius = [0.0, 0.0, None, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_mesh_grid2":
        edge_radius = [0.0, 0.0, None, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_mesh_grid-collider":
        edge_radius = [0.1, 0.08, None, 0.18, 0.2, 0.0, 0.18, 0.2, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_mesh_grid":
        edge_radius = [0.1, 0.0, None, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_mesh_grid2":
        edge_radius = [0.2, 0.0, None, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_equal_edges_t+1":
        edge_radius = [0.2, 0.08, None, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "coarse_equal_edges_t":
        edge_radius = [0.2, 0.08, None, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        input_timestep = "t"
        euclidian_distance = True
    # rev2 datasets
    elif dataset == "coarse_meshgraphnet_t":
        edge_radius = [0.0, 0.08, None, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_meshgraphnet_world_t":
        edge_radius = [0.0, 0.08, 0.35, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]  # 0.25 smallest mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_meshgraphnet_world_large_t":
        edge_radius = [0.0, 0.08, 0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3]  # 0.25 smallest mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_full_graph_t":
        edge_radius = [0.1, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_no_grid_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.3, 0.08, 0.08, 0.3]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_t":
        edge_radius = [0.0, 0.08, None, 0.08, 0.08, 0.0, 0.08, 0.08, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_larger_radius_t":
        edge_radius = [0.0, 0.08, None, 0.18, 0.2, 0.0, 0.18, 0.2, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "coarse_reduced_graph_larger_radius_t+1":
        edge_radius = [0.0, 0.08, None, 0.18, 0.2, 0.0, 0.18, 0.2, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    # tisue dataset
    elif dataset == "tissue_full_graph_t+1":
        edge_radius = [0.0035, 0.0, None, 0.008, 0.005, 0.02, 0.008, 0.005, 0.02]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "tissue_full_graph_t":
        edge_radius = [0.0035, 0.0, None, 0.008, 0.005, 0.02, 0.008, 0.005, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_sparse_full_graph_t+1":
        edge_radius = [0.0025, 0.0, None, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "tissue_sparse_full_graph_t":
        edge_radius = [0.0025, 0.0, None, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_sparse2_full_graph_t":
        edge_radius = [0.0035, 0.0, None, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_sparse_grid_full_graph_t+1":
        edge_radius = [0.0025, 0.0, None, 0.008, 0.005, 0.02, 0.008, 0.005, 0.02]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "tissue_sparse_grid_full_graph_t":
        edge_radius = [0.0025, 0.0, None, 0.008, 0.005, 0.02, 0.008, 0.005, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_sparse_full_graph_world_t":
        edge_radius = [0.0025, 0.0, 0.008, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_sparse_full_graph_world_large_t":
        edge_radius = [0.0025, 0.0, 0.01, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_no_grid_t+1":
        edge_radius = [0.0, 0.0, None, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "tissue_no_grid_t":
        edge_radius = [0.0, 0.0, None, 0.008, 0.003, 0.02, 0.008, 0.003, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_t+1":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_t":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_world_t":
        edge_radius = [0.0, 0.0, 0.008, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02]  # 0.0034 smalles mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_world_large_t":
        edge_radius = [0.0, 0.0, 0.01, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02]  # 0.0034 smalles mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_large_t+1":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_large_t":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_small_t":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_meshgraphnet_zero_t":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_dummy":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_reduced_graph_t+1":
        edge_radius = [0.0, 0.0, None, 0.008, 0.005, 0.0, 0.008, 0.005, 0.0]
        input_timestep = "t+1"
        euclidian_distance = True
    elif dataset == "tissue_reduced_graph_t":
        edge_radius = [0.0, 0.0, None, 0.008, 0.005, 0.0, 0.008, 0.005, 0.0]
        input_timestep = "t"
        euclidian_distance = True

    elif dataset == "tissue_norm_meshgraphnet_t":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_norm_meshgraphnet_world_t":
        edge_radius = [0.0, 0.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4]  # 0.0034 smalles mesh edge
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_norm_no_pcd_edges_t":
        edge_radius = [0.0, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_norm_full_graph_t":
        edge_radius = [0.07, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_norm_reduced_pcd_edges_t":
        edge_radius = [0.05, 0.0, None, 0.16, 0.06, 0.4, 0.16, 0.06, 0.4]
        input_timestep = "t"
        euclidian_distance = True


    elif dataset == "tissue_tube_meshgraphnet":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_tube_meshgraphnet_world":
        edge_radius = [0.0, 0.0, 0.07, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_tube_no_pcd_edges":
        edge_radius = [0.0, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_tube_no_pcd_edges_nc":
        edge_radius = [0.0, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_tube_full_graph":
        edge_radius = [0.05, 0.0, None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_timestep = "t"
        euclidian_distance = True
    elif dataset == "tissue_tube_no_collider":
        edge_radius = [0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        input_timestep = "t"
        euclidian_distance = True

    else:
        raise ValueError(f"Dataset {dataset} does currently not exist. Consider adding it to get_dataset_properties")

    edge_radius_dict = get_radius_dict(edge_radius)

    if "tissue" in dataset:
        tissue_task = True
    else:
        tissue_task = False

    return edge_radius_dict, input_timestep, euclidian_distance, tissue_task

#
