import torch
from torch_geometric.nn import MetaLayer

from util.Types import *
from modules.gnn_modules.AbstractMessagePassingBlock import AbstractMessagePassingBlock
from modules.gnn_modules.homogeneous_modules.MetaModules import EdgeModule, NodeModule, GlobalModule, GlobalModuleNoUpdate


class MessagePassingBlock(AbstractMessagePassingBlock):
    """
         Defines a single MessagePassingLayer that takes a homogeneous observation graph and updates its node and edge
         features using different modules (Edge, Node, Global).
         It first updates the edge-features. The node-features are updated next using the new edge-features. Finally,
         it updates the global features using the new edge- & node-features. The updates are done through MLPs.
         The three Modules (Edge, Node, Global) are combined into a MetaLayer.
    """
    def __init__(self, base_config: ConfigDict, in_node_features: int, in_edge_features: int,
                 in_global_features: int = 0, aggregation_function_str: str = "mean",
                 edge_base_config: Optional[ConfigDict] = None, node_base_config: Optional[ConfigDict] = None,
                 global_base_config: Optional[ConfigDict] = None, use_residual_connections: bool = False):
        """
        Args:
            base_config: Dictionary specifying the way that the gnn base should look like.
                It is passed on to the modules
            in_node_features: Dimension of each node feature vector
            in_edge_features: Dimension of each edge feature vector
            in_global_features: how many global features we have
            aggregation_function_str: How to aggregate the nodes/edges for the different modules.
                Must be some aggregation method such as "mean", "max" or "min".
            edge_base_config: Optional. If provided, will overwrite the mlp_config for the edge MLP
            node_base_config: Optional. If provided, will overwrite the mlp_config for the node MLPs
            global_base_config: Optional. If provided, will overwrite the mlp_config for the global MLP
            use_residual_connections: Whether to use residual connections for both nodes and edges or not. If True,
              the original inputs will be added to the outputs.
        """
        super().__init__(base_config, use_residual_connections)
        self.constant_global_features = bool(base_config.get("constant_global_features"))

        edge_module = EdgeModule(in_features=2 * in_node_features + in_edge_features + in_global_features,
                                 num_types=1,
                                 latent_dimension=in_edge_features,
                                 base_config=edge_base_config if edge_base_config is not None else base_config,
                                 aggregation_function_str=aggregation_function_str)
        self._out_edge_features = edge_module.out_features

        node_module = NodeModule(in_features=in_node_features + edge_module.out_features + in_global_features,
                                 num_types=1,
                                 latent_dimension=in_node_features,
                                 base_config=node_base_config if node_base_config is not None else base_config,
                                 aggregation_function_str=aggregation_function_str)
        self._out_node_features = node_module.out_features

        if in_global_features > 0:  # has global features
            global_base_config = global_base_config if global_base_config is not None else base_config
            in_features = node_module.out_features + in_global_features + edge_module.out_features

            if self.constant_global_features:
                global_module = GlobalModuleNoUpdate(in_features=in_features,
                                             num_types=1,
                                             latent_dimension=in_global_features,
                                             base_config=global_base_config,
                                             aggregation_function_str=aggregation_function_str)
            else:
                global_module = GlobalModule(in_features=in_features,
                                             num_types=1,
                                             latent_dimension=in_global_features,
                                             base_config=global_base_config,
                                             aggregation_function_str=aggregation_function_str)

            self._out_global_features = global_module.out_features
        else:
            global_module = None
            self._out_global_features = 0

        self._meta_layer = MetaLayer(edge_module, node_module, global_module)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor,
                global_features: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass for this message passing block

        Args:
            node_features: The features for each node of the graph. Has shape (num_nodes, num_features_per_node)
            edge_index: Connectivity Tensor of the graph. Has shape (2, num_edges)
            edge_features: Feature matrix of the edges. Has shape (num_edges, num_features_per_edge)
            global_features: Features for the whole graph. Has shape (num_global_features, num_graphs_in_batch)
            batch: Indexing for different graphs in the same badge.

        Returns:
            Updated node, edge and global features as a tuple

        """
        updated_node_features, updated_edge_features, updated_global_features = \
            self._meta_layer(node_features, edge_index, edge_features, global_features, batch)

        if self._use_residual_connections:  # simply add original features to updated ones
            updated_node_features = updated_node_features + node_features
            updated_edge_features = updated_edge_features + edge_features
            if global_features is not None:
                if self.constant_global_features:
                    updated_global_features = global_features
                else:
                    updated_global_features = updated_global_features + global_features

        return updated_node_features, updated_edge_features, updated_global_features
