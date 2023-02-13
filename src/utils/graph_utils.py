import torch_cluster
import torch
import torch_geometric.transforms as T

from util.Types import *
from src.utils.get_collider_triangles import get_tube_collider_triangles


def build_one_hot_features(num_per_type: list) -> Tensor:
    """
    Builds one-hot feature tensor indicating the edge/node type from numbers per type
    Args:
        num_per_type: List of numbers of nodes per type

    Returns:
        features: One-hot features Tensor
    """
    total_num = sum(num_per_type)
    features = torch.zeros(total_num, len(num_per_type))
    for typ in range(len(num_per_type)):
        features[sum(num_per_type[0:typ]): sum(num_per_type[0:typ+1]), typ] = 1
    return features


def build_type(num_per_type: list) -> Tensor:
    """
    Build node or edge type tensor from list of numbers per type
    Args:
        num_per_type: list of numbers per type

    Returns:
        features: Tensor containing the type as number
    """
    total_num = sum(num_per_type)
    features = torch.zeros(total_num)
    for typ in range(len(num_per_type)):
        features[sum(num_per_type[0:typ]): sum(num_per_type[0:typ+1])] = typ
    return features


def add_static_info(x: Tensor, mesh_positions: Tensor, num_per_node_type: list) -> Tensor:
    """
    Add one-hot encoding for (fixed) static nodes to the 2D data (bottom row)
    Args:
        mesh_positions: Tensor containing the mesh positions
        num_per_type: list of numbers per node type

    Returns:
        x: Updated node features with one-hot encoding
    """
    indices = torch.where(mesh_positions[:,1] == -1.0)
    static = torch.zeros_like(x[:,0]).view(-1,1)
    static[indices[0] + sum(num_per_node_type[0:2])] = 1
    x = torch.cat((x, static), dim=1)
    return x


def add_static_tissue_info(x: Tensor, num_per_node_type: list) -> Tensor:
    """
    Adds one-hot encoding for (fixed) static nodes to the (bottom face of the T part) tissue mesh
    Non-moving nodes ar assigned a zero, one else.
    Args:
        x: Node features
        num_per_type: list of numbers per node type

    Returns:
        x: Updated node features with one-hot encoding
    """
    if num_per_node_type[2] > 400:
        # only for tube task (750 nodes)  todo/quick and dirty solution
        indices = torch.linspace(0, 29, 30, dtype=int)
    else:
        indices = torch.tensor([59,  60,  61,  62,  84,  87, 105, 106, 169, 170, 171, 172, 194, 196,
            215, 216, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
            232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
            246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
            260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 273, 274, 275,
            276, 277, 280, 281, 310, 311, 312, 313, 314, 317, 318, 321, 322, 323,
            324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
            338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
            352, 353, 354, 355, 356, 357, 358, 359, 360])
    static = torch.ones_like(x[:,0]).view(-1,1)
    static[indices + sum(num_per_node_type[0:2])] = 0
    x = torch.cat((x, static), dim=1)
    return x


def get_relative_mesh_positions(mesh_edge_index: Tensor, mesh_positions: Tensor) -> Tensor:
    """
    Transform the positions of the mesh into a relative position encoding along with the Euclidean distance in the edges
    Args:
        mesh_edge_index: Tensor containing the mesh edge indices
        mesh_positions: Tensor containing mesh positions

    Returns:
        edge_attr: Tensor containing the batched edge features
    """
    data = Data(pos=mesh_positions,
                edge_index=mesh_edge_index)
    transforms = T.Compose([T.Cartesian(norm=False, cat=True), T.Distance(norm=False, cat=True)])
    data = transforms(data)
    return data.edge_attr


def add_relative_mesh_positions(edge_attr: Tensor, edge_type: Tensor, input_mesh_edge_index: Tensor, initial_mesh_positions: Tensor) -> Tensor:
    """
    Adds the relative mesh positions to the mesh edges (in contrast to the world edges) and zero anywhere else.
    Refer to MGN by Pfaff et al. 2020 for more details.
    Args:
        edge_attr: Current edge features
        edge_type: Tensor containing the edges types
        input_mesh_edge_index: Mesh edge index tensor
        initial_mesh_positions: Initial positions of the mesh nodes "mesh coordinates"

    Returns:
        edge_attr: updated edge features
    """
    indices = torch.where(edge_type == 2)[0]  # type 2: mesh edges
    mesh_edge_index = torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1).long()
    mesh_attr = get_relative_mesh_positions(mesh_edge_index, initial_mesh_positions)
    mesh_positions = torch.zeros(edge_attr.shape[0], mesh_attr.shape[1])
    mesh_positions[indices,:] = mesh_attr
    edge_attr = torch.cat((edge_attr, mesh_positions), dim=1)
    return edge_attr


def remove_duplicates_with_mesh_edges(mesh_edges: Tensor, world_edges: Tensor) -> Tensor:
    """
    Removes the duplicates with the mesh edges have of the world edges that are created using a nearset neighbor search. (only MGN)
    To speed this up the adjacency matrices are used
    Args:
        mesh_edges: edge list of the mesh edges
        world_edges: edge list of the world edges

    Returns:
        new_world_edges: updated world edges without duplicates
    """
    import torch_geometric.utils as utils
    adj_mesh = utils.to_dense_adj(mesh_edges)
    if world_edges.shape[1] > 0:
        adj_world = utils.to_dense_adj(world_edges)
    else:
        adj_world = torch.zeros_like(adj_mesh)
    if adj_world.shape[1] < adj_mesh.shape[1]:
        padding_size = adj_mesh.shape[1] - adj_world.shape[1]
        padding_mask = torch.nn.ConstantPad2d((0, padding_size, 0, padding_size), 0)
        adj_world = padding_mask(adj_world)
    elif adj_world.shape[1] > adj_mesh.shape[1]:
        padding_size = adj_world.shape[1] - adj_mesh.shape[1]
        padding_mask = torch.nn.ConstantPad2d((0, padding_size, 0, padding_size), 0)
        adj_mesh = padding_mask(adj_mesh)
    new_adj = adj_world-adj_mesh
    new_adj[new_adj < 0] = 0
    new_world_edges = utils.dense_to_sparse(new_adj)[0]
    return new_world_edges


def create_graph_from_raw(input_data,
                          edge_radius_dict: Dict,
                          output_device,
                          use_mesh_coordinates: bool = False,
                          hetero: bool = False,
                          tissue_task: bool = False):
    """
    Choose correct graph creation function for homogeneous or heterogeneous data
    """
    if hetero:
        return create_hetero_graph_from_raw(input_data,
                                            edge_radius_dict,
                                            output_device,
                                            use_mesh_coordinates,
                                            tissue_task)
    else:
        return create_homo_graph_from_raw(input_data,
                                          edge_radius_dict,
                                          output_device,
                                          use_mesh_coordinates,
                                          tissue_task)


def create_homo_graph_from_raw(input_data,
                               edge_radius_dict: Dict,
                               output_device,
                               use_mesh_coordinates: bool = False,
                               tissue_task: bool = False) -> Data:
    """
    Creates a homogeneous graph from the raw data (point cloud, collider, mesh) given the connectivity of the edge radius dict
    Args:
        input_data: Tuple containing the data for the time step
        edge_radius_dict: Edge radius dict describing the connectivity setting
        output_device: Working device for output data, either cpu or cuda
        use_mesh_coordinates: enables message passing also in mesh coordinate space
        tissue_task: True if 3D data is used

    Returns:
        data: Data element containing the built graph
    """
    if tissue_task:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, next_collider_positions, poisson_ratio = input_data
    else:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, poisson_ratio = input_data

    # dictionary for positions
    pos_dict = {'grid': grid_positions,
                'collider': collider_positions,
                'mesh': mesh_positions}

    # build nodes features (one hot)
    num_nodes = []
    for values in pos_dict.values():
        num_nodes.append(values.shape[0])
    x = build_one_hot_features(num_nodes)

    # add colors and collider velocity to point cloud (only 2D)
    if grid_colors is not None:
        x_colors = torch.zeros_like(x)
        x_colors[0:num_nodes[0], :] = grid_colors
        x_colors[num_nodes[0]:num_nodes[0] + num_nodes[1], 2] = torch.ones(num_nodes[1])*(-200.0/100*0.01)
        x = torch.cat((x, x_colors), dim=1)
    node_type = build_type(num_nodes)

    # # used if poisson ratio needed as input feature, but atm incompatible with Imputation training
    if poisson_ratio is not None:
        poisson_ratio = poisson_ratio.float()
        x_poisson = torch.ones_like(x[:, 0])
        x_poisson = x_poisson * poisson_ratio
        x = torch.cat((x, x_poisson.unsqueeze(1)), dim=1)
    else:
        poisson_ratio = torch.tensor([1.0])

    # for 3D data add collider velocity and static mesh node information to node features, poisson only used for 2D
    if tissue_task:
        collider_velocities = (next_collider_positions - collider_positions).squeeze()
        collider_velocities = torch.nn.functional.normalize(collider_velocities, dim=0)
        x_velocities = torch.zeros((x.shape[0], 3))
        x_velocities[num_nodes[0]:num_nodes[0]+num_nodes[1], :] = collider_velocities
        x = torch.cat((x, x_velocities), dim=1)
        x = add_static_tissue_info(x, num_nodes)

    # index shift dict for edge index matrix
    index_shift_dict = {'grid': 0,
                        'collider': num_nodes[0],
                        'mesh': num_nodes[0] + num_nodes[1]}
    # get gripper edges for tube task todo/quick and dirty solution
    # tube_task = False
    # if num_nodes[2] > 400:
    #     tube_task = True
    #     collider_edge_index = torch.tensor(get_tube_collider_triangles())

    # create edge_index dict with the same keys as edge_radius
    edge_index_dict = {}
    for key in edge_radius_dict.keys():
        if key[0] == key[2]:

            # use mesh connectivity instead of nearest neighbor
            if key[0] == 'mesh':
                edge_index_dict[key] = torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            # elif key[0] == 'collider':
            #     if tube_task:
            #         edge_index_dict[key] = torch.cat((collider_edge_index, collider_edge_index[[1, 0]]), dim=1)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            #     else:
            #         edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            # use radius graph for edges between nodes of the same type
            else:
                edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]

        # use radius for edges between different sender and receiver nodes
        else:
            edge_index_dict[key] = torch_cluster.radius(pos_dict[key[2]], pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            edge_index_dict[key][1, :] += index_shift_dict[key[2]]

    # add world edges if edge radius for mesh is not none
    mesh_key = ('mesh', '2', 'mesh')
    world_key = ('mesh', '9', 'mesh')
    if edge_radius_dict[mesh_key] is not None:
        edge_index_dict[world_key] = torch_cluster.radius_graph(pos_dict['mesh'], r=edge_radius_dict[mesh_key], max_num_neighbors=100)
        edge_index_dict[world_key][0, :] += index_shift_dict['mesh']
        edge_index_dict[world_key][1, :] += index_shift_dict['mesh']
        edge_index_dict[world_key] = remove_duplicates_with_mesh_edges(edge_index_dict[mesh_key], edge_index_dict[world_key])

    # build edge_attr (one-hot)
    num_edges = []
    for value in edge_index_dict.values():
        num_edges.append(value.shape[1])
    edge_attr = build_one_hot_features(num_edges)
    edge_type = build_type(num_edges)

    # add mesh_coordinates to mesh edges if used
    if use_mesh_coordinates:
        edge_attr = add_relative_mesh_positions(edge_attr, edge_type, input_mesh_edge_index, initial_mesh_positions)

    # create node positions tensor and edge_index from dicts
    pos = torch.cat(tuple(pos_dict.values()), dim=0)
    edge_index = torch.cat(tuple(edge_index_dict.values()), dim=1)

    # create data object for torch
    data = Data(x=x.float(),
                u=poisson_ratio,
                pos=pos.float(),
                edge_index=edge_index.long(),
                edge_attr=edge_attr.float(),
                y=label.float(),
                y_old=mesh_positions.float(),
                node_type=node_type,
                edge_type=edge_type,
                poisson_ratio=poisson_ratio).to(output_device)
    return data


def create_hetero_graph_from_raw(input_data,
                                 edge_radius_dict: Dict,
                                 output_device,
                                 use_mesh_coordinates: bool = False,
                                 tissue_task: bool = False) -> Data:
    """
    Creates a homogeneous graph from the raw data (point cloud, collider, mesh) given the connectivity of the edge radius dict which is then later converted to a heterogeneous graph by .to_heterogeneous
    Args:
        input_data: Tuple containing the data for the time step
        edge_radius_dict: Edge radius dict describing the connectivity setting
        output_device: Working device for output data, either cpu or cud
        use_mesh_coordinates: enables message passing also in mesh coordinate space (currently not supported for hetero)
        tissue_task: True if 3D data is used

    Returns:
        data: Data element containing the built graph
    """
    if tissue_task:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, next_collider_positions, poisson_ratio = input_data
    else:
        grid_positions, collider_positions, mesh_positions, input_mesh_edge_index, label, grid_colors, initial_mesh_positions, poisson_ratio = input_data

    # create node position dict
    pos_dict = {'grid': grid_positions,
                'collider': collider_positions,
                'mesh': mesh_positions}

    # build nodes features (one hot)
    num_nodes = []
    for values in pos_dict.values():
        num_nodes.append(values.shape[0])
    node_type = build_type(num_nodes)

    # index shift dict for edge index matrix
    index_shift_dict = {'grid': 0,
                        'collider': num_nodes[0],
                        'mesh': num_nodes[0] + num_nodes[1]}

    # get gripper edges for tube task todo/quick and dirty solution
    # tube_task = False
    # if num_nodes[2] > 400:
    #     tube_task = True
    #     collider_edge_index = torch.tensor(get_tube_collider_triangles())

    # create edge_index dict with the same keys as edge_radius
    edge_index_dict = {}
    for key in edge_radius_dict.keys():
        if key[0] == key[2]:

            # use mesh connectivity instead of nearest neighbor
            if key[0] == 'mesh':
                edge_index_dict[key] = torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            #
            # elif key[0] == 'collider':
            #     if tube_task:
            #         edge_index_dict[key] = torch.cat((collider_edge_index, collider_edge_index[[1, 0]]), dim=1)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]
            #     else:
            #         edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            #         edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            #         edge_index_dict[key][1, :] += index_shift_dict[key[2]]

            # use radius graph for edges between nodes of the same type
            else:
                edge_index_dict[key] = torch_cluster.radius_graph(pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
                edge_index_dict[key][0, :] += index_shift_dict[key[0]]
                edge_index_dict[key][1, :] += index_shift_dict[key[2]]

        # use radius for edges between different sender and receiver nodes
        else:
            edge_index_dict[key] = torch_cluster.radius(pos_dict[key[2]], pos_dict[key[0]], r=edge_radius_dict[key], max_num_neighbors=100)
            edge_index_dict[key][0, :] += index_shift_dict[key[0]]
            edge_index_dict[key][1, :] += index_shift_dict[key[2]]

        # add world edges if edge radius for mesh is not none
    mesh_key = ('mesh', '2', 'mesh')
    world_key = ('mesh', '9', 'mesh')
    if edge_radius_dict[mesh_key] is not None:
        edge_index_dict[world_key] = torch_cluster.radius_graph(pos_dict['mesh'], r=edge_radius_dict[mesh_key], max_num_neighbors=100)
        edge_index_dict[world_key][0, :] += index_shift_dict['mesh']
        edge_index_dict[world_key][1, :] += index_shift_dict['mesh']
        edge_index_dict[world_key] = remove_duplicates_with_mesh_edges(edge_index_dict[mesh_key], edge_index_dict[world_key])

    # build edge_attr (one-hot)
    num_edges = []
    for value in edge_index_dict.values():
        num_edges.append(value.shape[1])
    edge_type = build_type(num_edges)

    # create node pos tensor and edge_index from dicts
    pos = torch.cat(tuple(pos_dict.values()), dim=0)
    edge_index = torch.cat(tuple(edge_index_dict.values()), dim=1)

    if poisson_ratio is not None:
        x_poisson = torch.ones(pos.shape[0], 1)
        x_poisson = x_poisson * poisson_ratio
        x = x_poisson
    else:
        x = torch.ones(pos.shape[0], 1)
        poisson_ratio = torch.tensor([-1.0])

    # create data object for torch
    data = Data(x=x.float(),
                pos=pos.float(),
                edge_index=edge_index.long(),
                y=label.float(),
                y_old=mesh_positions.float(),
                node_type=node_type.long(),
                edge_type=edge_type.long(),
                poisson_ratio=poisson_ratio.view(1, -1),
                mesh_edges=torch.cat((input_mesh_edge_index, input_mesh_edge_index[[1, 0]]), dim=1).long().view(1, 2, -1))

    # add features to data which are later used after converting to a heterogeneous graph
    if grid_colors is not None:
        data.grid_colors = grid_colors.float()
    if tissue_task:
        collider_velocities = (next_collider_positions - collider_positions).squeeze()
        data.collider_velocities = torch.nn.functional.normalize(collider_velocities, dim=0).float()
        data.initial_mesh_positions = initial_mesh_positions.float().view(1, -1, 3)
    else:
        data.initial_mesh_positions = initial_mesh_positions.float().view(1, -1, 2)
    data.to(output_device)
    return data
