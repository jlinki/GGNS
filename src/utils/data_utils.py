import torch
from torch_geometric import transforms as T
from src.utils.graph_utils import add_static_tissue_info
from util.Types import *


def get_timestep_data(data: Data, predicted_position: Tensor, input_timestep: str, use_color: bool, use_poisson=False, tissue_task=False):
    """
    Function to get the correct data from a raw datalist containing point clouds, meshes and collider positions but no common graph
    Args:
        data: Raw PyG data element containing point cloud, mesh and collider positions and mesh edges
        predicted_position: Tensor containing the predicted mesh positions
        input_timestep: Str indicating if point cloud an collider are used from current time step 't' or 't+1'
        use_color: Use color gradient textured point cloud (only 2D)
        use_poisson: Use poisson ratio as feature
        tissue_task: True if 3D data is used

    Returns: Tuple.
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
    if input_timestep == "t":
        data_timestep = data.grid_positions_old, data.collider_positions_old, predicted_position.cpu(), data.mesh_edge_index, data.y
        if use_color:
            data_timestep += (data.grid_colors_old,)
        else:
            data_timestep += (None,)
        data_timestep += (data.initial_mesh_positions,)
        if tissue_task:
            data_timestep += (data.next_collider_positions_old,)
        if use_poisson:
            data_timestep += (data.poisson_ratio,)
        else:
            data_timestep += (None,)
    elif input_timestep == "t+1":
        data_timestep = data.grid_positions, data.collider_positions, predicted_position.cpu(), data.mesh_edge_index, data.y
        if use_color:
            data_timestep += (data.grid_colors,)
        else:
            data_timestep += (None,)
        data_timestep += (data.initial_mesh_positions,)
        if tissue_task:
            data_timestep += (data.next_collider_positions,)
        if use_poisson:
            data_timestep += (data.poisson_ratio,)
        else:
            data_timestep += (None,)
    return data_timestep


def get_relative_mesh_positions_batched(device, mesh_edge_index: Tensor, mesh_positions: Tensor) -> Tensor:
    """
    Transform the positions of the mesh into a relative position encoding along with the Euclidean distance in the edges for already batched input data
    Args:
        device: Used device, either 'cpu' or 'cuda'
        mesh_edge_index: Tensor containing a batch of mesh edge indices with shape (batch, 2, num_edges)
        mesh_positions: Tensor containing batched mesh positions

    Returns:
        edge_attr: Tensor containing the batched edge features with shape (batch*num_edges, F)
    """
    edge_attr_list = []
    for batch in range(mesh_edge_index.shape[0]):
        data = Data(pos=mesh_positions[batch, :],
                edge_index=mesh_edge_index[0, :]).to(device)
        transforms = T.Compose([T.Cartesian(norm=False, cat=True), T.Distance(norm=False, cat=True)])
        data = transforms(data)
        edge_attr_list.append(data.edge_attr)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    return edge_attr


def add_static_tissue_info_batched(device, x: Tensor, ptr: Tensor, old_node_type: list) -> Tensor:
    """
    Adds one-hot encoding for (fixed) static nodes to the (bottom face of the T part) tissue mesh for already batched input data
    Non-moving nodes ar assigned a zero, one else.
    Args:
        device: Used device, either 'cpu' or 'cuda'
        x: Node features
        ptr: Pointer on starting element of each batch

    Returns:
        x: Updated node features with one-hot encoding
    """

    if int(len(old_node_type[old_node_type == 2])/(len(ptr)-1)) > 400:
        # only for tube task todo quick and dirty solution
        indices = torch.linspace(0, 29, 30, dtype=int).to(device)
    else:
        indices = torch.tensor([59,  60,  61,  62,  84,  87, 105, 106, 169, 170, 171, 172, 194, 196,
            215, 216, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
            232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
            246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
            260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 273, 274, 275,
            276, 277, 280, 281, 310, 311, 312, 313, 314, 317, 318, 321, 322, 323,
            324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
            338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
            352, 353, 354, 355, 356, 357, 358, 359, 360]).to(device)
    static = torch.ones_like(x[:, 0]).view(-1,1).to(device)
    for ptr_idx in range(len(ptr)-1):
         static[indices + ptr[ptr_idx]] = 0
    x = torch.cat((x, static), dim=1)
    return x


def build_one_hot_features_batched(device, node_type: Tensor) -> Tensor:
    """
    Builds one-hot feature tensor indicating the node type for already batched input data
    Args:
        device: Used device, either 'cpu' or 'cuda'
        node_type: Tensor containing the node type index (0: point cloud, 1: collider, 2: mesh)

    Returns:
        features: One-hot features Tensor
    """
    features = torch.zeros((node_type.shape[0], torch.max(node_type).item() + 1)).to(device)
    features[node_type == 0, 0] = 1
    features[node_type == 1, 1] = 1
    features[node_type == 2, 2] = 1
    return features


def convert_to_mgn_hetero(data: Data, device, tissue_task: bool, use_mesh_coordinates: bool) -> HeteroData:
    """
    Converts the preprocessed homogeneous (batched) data element to a heterogeneous one with world and mesh edges (two edge types) and one node type.
    This follows the implementation of MGN by Pfaff et al. 2020
    Args:
        data: (batched) data element
        device: working device (cpu or cuda)
        tissue_task: (bool) Use setting for 3D deformable tissue task
        use_mesh_coordinates:  (bool) Use mesh coordinates in mesh edges (together with world coordinates)

    Returns: hetero data element
    """
    old_edge_type = data.edge_type
    old_node_type = data.node_type
    new_edge_type = torch.ones_like(old_edge_type)
    mesh_edges_idx = old_edge_type == 2
    new_edge_type[mesh_edges_idx] = 0
    data.edge_type = new_edge_type.to(device)
    data.node_type = torch.zeros_like(old_node_type).to(device)
    edge_type_name = [('mesh', '0', 'mesh'), ('mesh', '1', 'mesh')]

    # Convert to hetero data, for only one node-type, all other attributes are added to its node stores
    hetero_data = data.to_heterogeneous(node_type_names=['mesh'], edge_type_names=edge_type_name)

    # Retrieve data from node_stores
    hetero_data.y = hetero_data.node_stores[0].y
    hetero_data.y_old = hetero_data.node_stores[0].y_old
    hetero_data.node_type = old_node_type

    # build one-hot node encoding
    hetero_data.node_stores[0].x = torch.cat((hetero_data.node_stores[0].x, build_one_hot_features_batched(device, old_node_type)), dim=1)

    if tissue_task:
        # add gripper velocity to gripper nodes
        hetero_data.node_stores[0].x = torch.cat((hetero_data.node_stores[0].x, build_one_hot_features_batched(device, old_node_type)), dim=1)
        x_velocities = torch.zeros((hetero_data.node_stores[0].x.shape[0], 3)).to(device)
        x_velocities[old_node_type == 1, :] = data.collider_velocities.view(-1, 3)
        hetero_data.node_stores[0].x = torch.cat((hetero_data.node_stores[0].x, x_velocities), dim=1)
        # add static mesh nodes info to mesh nodes
        hetero_data.node_stores[0].x = add_static_tissue_info_batched(device, hetero_data.node_stores[0].x, data.ptr, old_node_type)

    # add mesh coordinates to ('mesh', '1', 'mesh') edges
    if use_mesh_coordinates:
        mesh_attr = get_relative_mesh_positions_batched(device, data.mesh_edges, data.initial_mesh_positions)
        hetero_data.edge_stores[0].edge_attr = torch.cat((hetero_data.edge_stores[0].edge_attr, mesh_attr), dim=1)

    return hetero_data


def convert_to_hetero_data(data: Data, hetero: bool, use_color: bool, device, tissue_task: bool, use_world_edges: bool, use_mesh_coordinates: bool, mgn_hetero=False):
    """

    Args:
        data: Homogeneous data element
        hetero: Does nothing if False
        use_color: Color gradient texture for point cloud
        device: Working device, either cpu or cuda
        tissue_task: True if 3D data is used
        use_world_edges: Use of explicit world edges (see MGN)
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        mgn_hetero: Use small hetero GNN like in MGN

    Returns:
        hetero_data: HeteroData or Data object depending on 'hetero'

    """
    if hetero:
        edge_type_name = [('grid', '0', 'grid'),
                          ('collider', '1', 'collider'),
                          ('mesh', '2', 'mesh'),
                          ('collider', '3', 'grid'),
                          ('mesh', '4', 'grid'),
                          ('mesh', '5', 'collider'),
                          ('grid', '6', 'collider'),
                          ('grid', '7', 'mesh'),
                          ('collider', '8', 'mesh')]
        if use_world_edges:
            edge_type_name.extend([('mesh', '9', 'mesh'),
                          ('collider', '10', 'collider'),
                          ('grid', '11', 'grid')])

        # convert to hetero dataset, using all edge and node types or only 2 edges type for mgn_hetero
        if mgn_hetero:
            hetero_data = convert_to_mgn_hetero(data, device, tissue_task, use_mesh_coordinates)
        else:
            hetero_data = data.to_heterogeneous(node_type_names=['grid', 'collider', 'mesh'],
                                                edge_type_names=edge_type_name)
            if use_color:
                hetero_data.node_stores[0].x = torch.cat((hetero_data.node_stores[0].x, data.grid_colors.float()), dim=1)
                collider_velocity = torch.ones(hetero_data.node_stores[1].x.shape[0]).to(device) * (-200.0 / 100 * 0.01)
                hetero_data.node_stores[1].x = torch.cat((hetero_data.node_stores[1].x, collider_velocity.view(-1, 1)), dim=1)
            if tissue_task:
                # add gripper velocity to gripper nodes
                hetero_data.node_stores[1].x = torch.cat((hetero_data.node_stores[1].x, data.collider_velocities.view(-1, 3)), dim=1)
                # add static mesh nodes info to mesh nodes
                hetero_data.node_stores[2].x = add_static_tissue_info(hetero_data.node_stores[2].x, [0, 0, 0])

            # add mesh coordinates to ('mesh', '2', 'mesh') edges
            if use_mesh_coordinates:
                mesh_attr = get_relative_mesh_positions_batched(device, data.mesh_edges, data.initial_mesh_positions).to(device)
                hetero_data.edge_stores[2].edge_attr = torch.cat((hetero_data.edge_stores[2].edge_attr, mesh_attr), dim=1)
    else:
        hetero_data = data

    return hetero_data


def get_feature_info_from_data(data_point: Data, device, hetero: bool, use_color: bool, tissue_task: bool, use_world_edges: bool, use_mesh_coordinates: bool, mgn_hetero: bool):
    """
    Retrieves the features dimensions from the generated data
    Args:
        data_point: Single PyG data element from dataset
        device: Working device, either cpu or cuda
        hetero: Does nothing if False
        use_color: Color gradient texture for point cloud
        tissue_task: True if 3D data is used
        use_world_edges: Use of explicit world edges (see MGN)
        use_mesh_coordinates: Encode mesh coordinate into mesh edges
        mgn_hetero: Use small hetero GNN like in MGN

    Returns: Tuple.
        in_node_features: Dictionary of node types
        in_edge_features: Dictionary of edge types
        num_node_features: Feature dimension of node features
        num_edge_features: Feature dimension of edge features

    """
    if hetero:
        data_point.batch = torch.zeros_like(data_point.x)
        data_point.ptr = torch.tensor([0, data_point.batch.shape[0]])
        data_point = convert_to_hetero_data(data_point, hetero, use_color, device, tissue_task, use_world_edges, use_mesh_coordinates, mgn_hetero)
        if mgn_hetero:
            num_node_features = data_point.node_stores[0].num_features
            num_edge_features = data_point.edge_stores[0].num_edge_features
            in_node_features = {'mesh': data_point.node_stores[0].num_features}
            in_edge_features = {('mesh', '0', 'mesh'): data_point.edge_stores[0].num_edge_features,
                            ('mesh', '1', 'mesh'): data_point.edge_stores[1].num_edge_features}
        else:
            num_node_features = data_point.node_stores[0].num_features
            num_edge_features = data_point.edge_stores[0].num_edge_features
            in_node_features = {'grid': data_point.node_stores[0].num_features,
                                'collider': data_point.node_stores[1].num_features,
                                'mesh': data_point.node_stores[2].num_features}
            in_edge_features = {('grid', '0', 'grid'): data_point.edge_stores[0].num_edge_features,
                            ('collider', '1', 'collider'): data_point.edge_stores[1].num_edge_features,
                            ('mesh', '2', 'mesh'): data_point.edge_stores[2].num_edge_features,
                            ('collider', '3', 'grid'): data_point.edge_stores[3].num_edge_features,
                            ('mesh', '4', 'grid'): data_point.edge_stores[4].num_edge_features,
                            ('mesh', '5', 'collider'): data_point.edge_stores[5].num_edge_features,
                            ('grid', '6', 'collider'): data_point.edge_stores[6].num_edge_features,
                            ('grid', '7', 'mesh'): data_point.edge_stores[7].num_edge_features,
                            ('collider', '8', 'mesh'): data_point.edge_stores[8].num_edge_features}
            if len(data_point.edge_stores) > 9:
                in_edge_features[('mesh', '9', 'mesh')] = data_point.edge_stores[9].num_edge_features
                in_edge_features[('collider', '10', 'collider')] = data_point.edge_stores[0].num_edge_features
                in_edge_features[('grid', '11', 'grid')] = data_point.edge_stores[0].num_edge_features
    else:
        in_node_features = {"0": data_point.x.shape[1]}
        in_edge_features = {"0": data_point.edge_attr.shape[1]}
        num_node_features = in_node_features['0']
        num_edge_features = in_edge_features['0']

    return in_node_features, in_edge_features, num_node_features, num_edge_features


def get_shortest_trajectory(trajectory_list: list) -> int:
    """
    Calculates the length of the shortest trajectory in a list of trajectories
    Args:
        trajectory_list: List of trajectories
    Returns:
        minimum: Minimum length of trajectory
    """
    minimum = len(trajectory_list[0])
    for trajectory in trajectory_list:
        min_i = len(trajectory)
        if min_i < minimum:
            minimum = min_i
    return minimum


def count_parameters(model, show_details: bool) -> int:
    """
    Counts the parameters of a torch.nn.Module model and outputs their names and paramter sizes if needed
    Args:
        model: torch.nn.Module model
        show_details: Print detailed information about layers

    Returns:
        total_params: Overall number of paramters in model
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        if show_details:
            print(name)
            print(param)
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    return total_params


def convert_trajectory_to_data_list(trajectory_list: list, start_index=0) -> list:
    """
    Converts a list of trajectories (list of time step data) to a single sequential list of all time steps
    Args:
        trajectory_list: List of trajectories
        start_index: Where to start a trajectory default: 0, at the beginning
    Returns:
        data_list: One list of all time steps
    """
    data_list = []
    for trajectory in trajectory_list:
        for index, data in enumerate(trajectory):
            if index >= start_index:
                data_list.append(data)

    return data_list


def crop_list_of_trajectories(traj_list: list, start: int, stop: int) -> list:
    """
    Crops the trajectories in a list of trajectories
    Args:
        traj_list: List of trajectories
        start: Where to start a trajectory
        stop: Where to end a trajectory
    Returns:
        data_list: One list of all time steps
    """
    traj_list_crop = []
    for index, trajectory in enumerate(traj_list):
        traj_list_crop.append(trajectory[start:stop])
    return traj_list_crop


def predict_velocity(node_features: Tensor, data: Data, hetero: bool, mgn_hetero: bool) -> Tensor:
    """
    Get velocities (or other node-wise feedback) for the batched mesh nodes from all output node features
    Args:
        node_features: Tensor containing decodes node features
        data: Input data element
        hetero: Is heterogeneous data used
        mgn_hetero: Is the MGN hetero data used

    Returns:
        mesh_velocity: Tensor containing the velocities of the mesh nodes
    """
    if hetero:
        if mgn_hetero:
            indices = torch.where(data.node_type == 2)[0]  # type 2: mesh nodes
            mesh_velocity = node_features[indices]
        else:
            mesh_velocity = node_features
    else:
        indices = torch.where(data.node_type == 2)[0]  # type 2: mesh nodes
        mesh_velocity = node_features[indices]

    return mesh_velocity


def transform_position_to_edges(data: Data, euclidian_distance: bool) -> Data:
    """
    Transform the node positions in a homogeneous data element to the edges as relative distance together with (if needed) Euclidean norm
    Args:
        data: Data element
        euclidian_distance: True if Euclidean norm included as feature

    Returns:
        out_data: Transformed data object
    """
    if euclidian_distance:
        data_transform = T.Compose([T.Cartesian(norm=False, cat=True), T.Distance(norm=False, cat=True)])
    else:
        data_transform = T.Compose([T.Cartesian(norm=False, cat=True)])
    out_data = data_transform(data)
    return out_data
