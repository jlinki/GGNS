import random
import numpy as np
import torch
from torch import nn as nn
from tqdm import tqdm
import wandb

from util.Types import *


def get_network_config(config: Dict, hetero: bool, use_world_edges: bool) -> Dict:
    """
    Load the corresponding entries of the config dict into the network config dict used by the GNN_Base
    Args:
        config: Input dictionary containing all parameters
        hetero: (bool) use hetero GNN
        use_world_edges: True if world edges are used

    Returns:
        network dict: Dictionary containing the network parameters
    """
    network_config = {"aggregation_function": config.get("aggregation_function"),
                          "latent_dimension": config.get("latent_dimension"),
                            "mlp_decoder": config.get("mlp_decoder"),
                          "base": {
                              "use_global_features": bool(config.get("use_global_features")),
                              "num_blocks": config.get("num_blocks"),
                              "use_residual_connections": config.get("use_residual_connections"),

                              "mlp": {
                                  "activation_function": config.get("activation_function"),
                                  "num_layers": config.get("num_layers"),
                                  "output_layer": config.get("output_layer"),
                                  "regularization": {
                                      "latent_normalization": config.get("latent_normalization"),
                                      "dropout": config.get("dropout")
                                  }
                              }
                          }
                          }
    if hetero:
        hetero_config = {"het_neighbor_aggregation": config.get("het_neighbor_aggregation"),
                         "het_edge_shared_weights": config.get("het_edge_shared_weights"),
                         "het_node_shared_weights": config.get("het_node_shared_weights"),
                         "het_world_edges": bool(use_world_edges)}
        network_config['base'].update(hetero_config)
    return network_config


def log_gradients(GNN, wandb_log: bool, epoch: int, eval_log_interval: int, hetero: bool):
    """
    Logs the mean gradient of the first, the last and all layers of our GNN using wandb
    Args:
        GNN:
        wandb_log: (bool) logging activated
        epoch: logging epoch
        eval_log_interval: logging interval for evaluation
        hetero: (bool) use hetero GNN
    """
    if wandb_log:
        if hetero:
            pass
        else:
            if epoch % eval_log_interval == 0:
                grad_first_layer = calculate_gradients_for_MP_layer(GNN, 0)
                grad_last_layer = calculate_gradients_for_MP_layer(GNN, -1)
                grad = []
                for param in GNN.parameters():
                    if param.grad is not None:
                        grad.append(param.grad.view(-1))
                grad = torch.cat(grad).abs().mean()
                wandb.log({"mean absolut gradient/first MP-layer": grad_first_layer, "epoch": epoch})
                wandb.log({"mean absolut gradient/last MP-layer": grad_last_layer, "epoch": epoch})
                wandb.log({"mean absolut gradient/all layers": grad, "epoch": epoch})


def calculate_gradients_for_MP_layer(GNN, layer):
    """
    Calculates the mean gradient of our GNN for a given layer
    Args:
        GNN: gnn_base object
        layer: layer number

    Returns:
        Mean gradient for the layer
    """
    grad = []
    for name, edge_parameter in GNN._gnn_base._message_passing_stack._blocks[layer]._meta_layer.edge_model._out_mlp.feedforward_layers.named_parameters():
        grad.append(edge_parameter.grad.view(-1))
    for name, node_parameter in GNN._gnn_base._message_passing_stack._blocks[layer]._meta_layer.node_model._out_mlp.feedforward_layers.named_parameters():
        grad.append(node_parameter.grad.view(-1))
    grad = torch.cat(grad).abs().mean()

    return grad


def seed_worker(worker_id):
    """
    Seeds the worker for the torch dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_all(seed):
    """
    Seeds all torch processes with seed: seed
    Args:
        seed: seed to use, default is 42

    """
    if not seed:
        seed = 42

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_noise_to_mesh_nodes(data: Data, sigma: float, device):
    """
    Adds training noise to the mesh node positions with standard deviation sigma
    Args:
        data: PyG data element containing (a batch of) graph(s)
        sigma: standard deviation of used noise
        device: working device (cuda or cpu)

    Returns:
        data: updated graph with noise

    """
    if sigma > 0.0:
            indices = torch.where(data.node_type == 2)[0]
            num_noise_features = data.pos.shape[1]
            num_node_features = data.pos.shape[1]
            noise = (torch.randn(indices.shape[0], num_noise_features) * sigma).to(device)
            data.pos[indices, num_node_features-num_noise_features:num_node_features] = data.pos[indices, num_node_features-num_noise_features:num_node_features] + noise
            data.y_old = data.y_old + noise
    return data


def add_noise_to_pcd_points(data: Data, sigma: float, device):
    """
    Adds training noise to the point cloud positions with standard deviation sigma
    Args:
        data: PyG data element containing (a batch of) graph(s)
        sigma: standard deviation of used noise
        device: working device (cuda or cpu)

    Returns:
        data: updated graph with noise

    """
    if sigma > 0.0:
            indices = torch.where(data.node_type == 0)[0]
            num_noise_features = data.pos.shape[1]
            num_node_features = data.pos.shape[1]
            noise = (torch.randn(indices.shape[0], num_noise_features) * sigma).to(device)
            data.pos[indices, num_node_features-num_noise_features:num_node_features] = data.pos[indices, num_node_features-num_noise_features:num_node_features] + noise
    return data


def add_pointcloud_dropout(data: Data, pointcloud_dropout: float, hetero: bool, use_world_edges=False) -> Data:
    """
    Randomly drops the pointcloud (with nodes and edges) for the input batch. A bit hacky
    data.batch and data.ptr are used
    Args:
        data: PyG data element containing (a batch of) heterogeneous or homogeneous graph(s)
        pointcloud_dropout: Probability of dropping the point cloud for a batch
        hetero: Use hetero data
        use_world_edges: Use world edges

    Returns:
        data: updated data element
    """
    x = np.random.rand(1)
    if x < pointcloud_dropout:
        # node and edge types to keep
        node_types = [1, 2]
        if use_world_edges:
            edge_types = [1, 2, 5, 8, 9]
        else:
            edge_types = [1, 2, 5, 8]

        # extract correct edge indices
        edge_indices = []
        for edge_type in edge_types:
            edge_indices.append(torch.where(data.edge_type == edge_type)[0])
        edge_indices = torch.cat(edge_indices, dim=0)

        # create index shift lists for edge index
        num_node_type = []
        num_node_type_0 = []
        graph_pointer = []
        for batch in range(int(torch.max(data.batch) + 1)):
            batch_data = data.node_type[data.batch == batch]
            num_node_type_0.append(len(batch_data[batch_data == 0]))
            graph_pointer.append(len(batch_data[batch_data == 1]) + len(batch_data[batch_data == 2]))
            num_node_type.append(len(batch_data))

        num_node_type_0 = list(np.cumsum(num_node_type_0))
        num_node_type = list(np.cumsum(num_node_type))
        num_node_type = [0] + num_node_type
        graph_pointer = [0] + list(np.cumsum(graph_pointer))

        # extract correct node indices (in batch order)
        # therefore the index shift list num_node_type is needed
        # to_heterogeneous does not care about batch indices, so to make this work, we need to keep the order of the batch when extracting the mesh only data
        node_indices = []
        for batch in range(int(torch.max(data.batch) + 1)):
            batch_data = data.node_type[data.batch == batch]
            for node_type in node_types:
                node_indices.append(torch.where(batch_data == node_type)[0] + num_node_type[batch])
        node_indices = torch.cat(node_indices, dim=0)

        # create updated tensors
        new_pos = data.pos[node_indices]
        new_x = data.x[node_indices]
        new_batch = data.batch[node_indices]
        new_node_type = data.node_type[node_indices]
        new_edge_index = data.edge_index[:,edge_indices]
        new_edge_type = data.edge_type[edge_indices]

        # shift indices for updated edge_index tensor:
        for index in range(len(num_node_type_0)):
            new_edge_index = torch.where(torch.logical_and(new_edge_index > num_node_type[index],  new_edge_index < num_node_type[index+1]), new_edge_index - num_node_type_0[index], new_edge_index)

        # update data object
        data.pos = new_pos
        data.x = new_x
        data.batch = new_batch
        data.node_type = new_node_type
        data.edge_index = new_edge_index
        data.edge_type = new_edge_type
        data.ptr = torch.tensor(graph_pointer)

        # edge_attr are only used for homogeneous graphs at this stage
        if not hetero:
            new_edge_attr = data.edge_attr[edge_indices]
            data.edge_attr = new_edge_attr

    return data


def joint_shuffle(*args):
    """
    Jointly shuffles all given torch tensors with the same length and possibly different dimensions
    Args:
        args: A number of tensors

    Returns:
        A random permutation of all tensors. E.g.
        ([a2, a1, a4, a3], [b2, b1, b4, b3], [c2, c1, c4, c3], ...)
    """
    first_tensor = args[0]
    assert all(len(first_tensor) == len(other_tensor) for other_tensor in args), "All arrays must have same length"
    permutation = torch.randperm(len(first_tensor))
    return (tensor[permutation] for tensor in args)


def apply_input_dropout(data: Data, dropout: float, hetero: bool) -> Data:
    """
    Applies edge dropout to the input graph (batch of graphs)
    Args:
        data: PyG data element containing (a batch of) graph(s)
        dropout: dropout probabilities [0, 1.0]
        hetero: (bool) hetero datatype

    Returns:
        The input graph with reduced connectivity/less edges

    """
    if dropout > 0.0:
        if hetero:
            for index, edge_name in enumerate(data.edge_types):
                kept_features = int((1.0 - dropout) * data.edge_stores[index].edge_index.shape[1])
                data.edge_stores[index].edge_attr, data.edge_stores[index].edge_index = [x[:kept_features] for x in joint_shuffle(data.edge_stores[index].edge_attr, data.edge_stores[index].edge_index.t())]
                data.edge_stores[index].edge_index = data.edge_stores[index].edge_index.t()
        else:
            kept_features = int((1.0 - dropout)*data.edge_index.shape[1])
            data.edge_attr, data.edge_index, data.edge_type = [x[:kept_features] for x in joint_shuffle(data.edge_attr, data.edge_index.t(), data.edge_type)]
            data.edge_index = data.edge_index.t()
    return data


def calculate_loss_normalizer(loader, device) -> float:
    """
    Calculates the average loss between two data samples in the data set as an optional loss normalizer
    A normalized loss of 1 would then correspond to the average loss between two data samples in the training set
    Args:
        loader: Dataloader object, containing the data
        device: device used (cuda or cpu)

    Returns:
        time_diff_loss: The normalizer

    """
    time_diff_loss = 0
    for data in tqdm(loader):
        data.to(device)
        criterion = nn.MSELoss()
        loss = criterion(data.y, data.y_old)
        time_diff_loss += loss
    time_diff_loss = time_diff_loss.item()/len(loader)

    return time_diff_loss
