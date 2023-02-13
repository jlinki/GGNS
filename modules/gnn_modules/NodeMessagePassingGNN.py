from torch import nn as nn
import torch
from torch.nn import utils
from modules.ModuleUtility import get_activation_and_regularization_layers, get_layer_size_layout
import numpy as np
from util.Types import *
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.data.data import Data
from modules.HelperModules import SkipConnection
from torch_scatter import scatter_mean

def _build_message_passing_module(num_node_features: int,
                                  message_passing_config: ConfigDict,
                                  out_features: Optional[int] = None) -> Tuple[nn.ModuleList, nn.ModuleList,
                                                                               nn.ModuleList, Optional[nn.Module]]:
    """
    Builds the discriminator (sub)network. This part of the network accepts some latent space as the input and
    outputs a classification
    Args:
        num_node_features: Number of input features
        message_passing_config: Dictionary containing the specification for the message passing module.
          Includes num_layers, layer_size and the type of layer to build

    Returns: A nn.ModuleList representing the graph network module

    """
    assert isinstance(num_node_features, (int, np.int32, np.int64)), f"Need to provide an integer number of " \
                                                                     f"features. Got '{num_node_features}' of type" \
                                                                     f"'{type(num_node_features)}' instead."

    regularization_config = message_passing_config.get("regularization")

    spectral_norm: bool = regularization_config.get("spectral_norm")  # todo test if this works with GCNs
    layer_type: str = message_passing_config.get("layer_type")
    use_skip_connections: bool = message_passing_config.get("use_skip_connections")
    network_layout = get_layer_size_layout(max_neurons=message_passing_config.get("max_neurons"),
                                           num_layers=message_passing_config.get("num_layers"),  # hidden layers
                                           network_shape=message_passing_config.get("network_shape"))
    message_passing_layers = nn.ModuleList()
    auxiliary_modules = nn.ModuleList()
    skip_connection_modules = nn.ModuleList()

    previous_shape = num_node_features
    in_channels = num_node_features
    for current_layer_size in network_layout:  # define message passing layer
        # todo add options

        message_passing_layer = get_message_passing_layer(layer_type=layer_type,
                                                          in_channels=in_channels,
                                                          out_channels=current_layer_size)

        # add layer to moduleList
        if spectral_norm:
            message_passing_layers.append(utils.spectral_norm(message_passing_layer))
        else:
            message_passing_layers.append(message_passing_layer)

        additional_layers = get_activation_and_regularization_layers(in_features=current_layer_size,
                                                                     regularization_config=regularization_config)
        auxiliary_modules.append(additional_layers)

        if use_skip_connections:
            skip_connection_layer = SkipConnection()
            skip_connection_modules.append(skip_connection_layer)
            in_channels = current_layer_size + previous_shape
            previous_shape = current_layer_size
        else:
            in_channels = current_layer_size
            previous_shape = current_layer_size

    if out_features:
        out_layer = get_message_passing_layer(layer_type=layer_type, in_channels=in_channels,
                                              out_channels=out_features)
    else:
        out_layer = None

    return message_passing_layers, auxiliary_modules, skip_connection_modules, out_layer


def get_message_passing_layer(layer_type: str, in_channels: int, out_channels: int):
    if layer_type.lower() == "gcn":
        message_passing_layer = GCNConv(in_channels=in_channels, out_channels=out_channels)
    elif layer_type.lower() == "gatv2":
        message_passing_layer = GATv2Conv(in_channels=in_channels, out_channels=out_channels)
    else:
        raise NotImplementedError(f"GNN layer '{layer_type}' not implemented")
    return message_passing_layer


class NodeMessagePassingGNN(nn.Module):
    """
    Message Passing module. Expects a graph as an input and performs a permutation equivariant operation followed
     by some activation and regularization on it. Outputs some input x and computes an output f(x).
    """

    def __init__(self, num_node_features: int, message_passing_config: ConfigDict,
                 out_features: Optional[int] = None, aggregate_nodes: bool=False):
        """
        Initializes a new encoder module that consists of multiple different layers that together encode some
        input into a latent space
        Args:
            num_node_features: Number of features per node
            message_passing_config: Dict containing information about what kind of message passing layer to build.
            Includes a regularization_config containing information about batchnorm, spectral norm and dropout
            out_features: Number of output dimensions. This corresponds to e.g., the number of classes or number of
              dimensions to regress to. If this value is not set, the hidden state of the (activated) last message
              passing layer will be returned instead
            aggregate_nodes: Whether to aggregate the nodes into a graph-wise feature vector or not
        """
        super().__init__()
        modules = _build_message_passing_module(num_node_features=num_node_features,
                                                message_passing_config=message_passing_config,
                                                out_features=out_features)

        if aggregate_nodes:
            self._aggregation_function = scatter_mean
        else:
            self._aggregation_function = None  # do not aggregate

        self.message_passing_layers, self.auxiliary_modules, self.skip_connections, self.out_layer = modules

    def forward(self, graph_batch: Data, return_latent_features: bool = False) -> Union[torch.Tensor,
                                                                                        Tuple[torch.Tensor,
                                                                                              torch.Tensor]]:
        """
        Computes the forward pass for the given input graph
        Args:
            graph_batch: Some input tensor x
            return_latent_features: Whether to also return latent features of the last regular layer or not

        Returns: The processed tensor h^l=f(x), where h^l is the

        """
        node_features = graph_batch.x if graph_batch.x is not None else graph_batch.pos
        edge_indices = graph_batch.edge_index
        previous_features = node_features
        for layer_index, (message_passing_layer, auxiliary_module) in enumerate(zip(self.message_passing_layers,
                                                                                    self.auxiliary_modules)):
            new_node_features = message_passing_layer(node_features, edge_indices)
            for auxiliary_layer in auxiliary_module:  # activations, dropout, batch_norm...
                new_node_features = auxiliary_layer(new_node_features)
            if self.skip_connections:
                node_features = self.skip_connections[layer_index](new_node_features, previous_features)
                previous_features = new_node_features
            else:
                node_features = new_node_features

        if self.out_layer:  # only apply if not empty
            out_features = self.out_layer(node_features, edge_indices)
        else:
            out_features = node_features

        out_features = out_features.squeeze(dim=1)

        if self._aggregation_function:  # potentially aggregate output over nodes for each graph
            out_features = self._aggregation_function(out_features, graph_batch.batch)

        # aggregate last hidden features for all nodes for each graph
        node_features = scatter_mean(node_features, graph_batch.batch)
        if return_latent_features:
            return out_features, node_features
        else:
            return out_features
