from modules.gnn_modules.AbstractMessagePassingBlock import AbstractMessagePassingBlock
from util.Types import *
import util.Keys as Keys

from modules.gnn_modules.heterogeneous_modules.HeteroMetaModules\
    import HeteroEdgeModule, HeteroNodeModule, HeteroGlobalModule
from modules.gnn_modules.heterogeneous_modules.HeteroMetaLayer import HeteroMetaLayer


class HeteroMessagePassingBlock(AbstractMessagePassingBlock):
    """
         Defines a single MessagePassingLayer that takes a heterogeneous observation graph and updates its node and edge
         features using different modules (Edge, Node, Global).
         It first updates the edge-features. The node-features are updated next using the new edge-features. Finally,
         it updates the global features using the new edge- & node-features. The updates are done through MLPs.
         The three Modules (Edge, Node, Global) are combined into a heterogeneous MetaLayer.
    """
    def __init__(self, base_config: ConfigDict, in_node_features: int, num_node_types: int, in_edge_features: int,
                 num_edge_types: int, in_global_features: int = 0, aggregation_function_str: str = "mean",
                 edge_base_config: Optional[ConfigDict] = None, node_base_config: Optional[ConfigDict] = None,
                 global_base_config: Optional[ConfigDict] = None, use_residual_connections: bool = False):
        """
        Args:
            base_config: Dictionary specifying the way that the gnn base should look like.
                het_neighbor_aggregation: If we want to concatenate the different edge types or aggregate them.
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
        self._het_neighbor_aggregation: str = base_config.get("het_neighbor_aggregation")

        # edge module:
        edge_module = HeteroEdgeModule(in_features=2 * in_node_features + in_edge_features + in_global_features,
                                       num_types=num_edge_types,
                                       latent_dimension=in_edge_features,
                                       base_config=edge_base_config if edge_base_config is not None else base_config,
                                       aggregation_function_str=aggregation_function_str)
        self._out_edge_features = edge_module.out_features

        # node module:
        in_features = in_node_features + edge_module.out_features + in_global_features
        if base_config.get("het_neighbor_aggregation") == Keys.CONCAT_AGGR:
            # here we do not aggregate types, so we need more in_features
            if base_config.get("het_world_edges"):
                in_features += edge_module.out_features * (num_node_types)
            else:
                in_features += edge_module.out_features * (num_node_types-1)

        node_module = HeteroNodeModule(in_features=in_features,
                                       num_types=num_node_types,
                                       latent_dimension=in_node_features,
                                       base_config=node_base_config if node_base_config is not None else base_config,
                                       aggregation_function_str=aggregation_function_str)
        self._out_node_features = node_module.out_features

        # global module:
        if in_global_features > 0:  # has global features
            global_base_config = global_base_config if global_base_config is not None else base_config
            in_features = node_module.out_features + in_global_features + edge_module.out_features
            if self._het_neighbor_aggregation == Keys.CONCAT_AGGR:
                # here we do not aggregate types, so we need more in_features
                in_features += node_module.out_features * (num_node_types - 1) +\
                               edge_module.out_features * (num_edge_types - 1)

            global_module = HeteroGlobalModule(in_features=in_features,
                                               num_types=1,
                                               latent_dimension=in_global_features,
                                               base_config=global_base_config,
                                               aggregation_function_str=aggregation_function_str)
            self._out_global_features = global_module.out_features
        else:
            global_module = None
            self._out_global_features = 0

        self._meta_layer = HeteroMetaLayer(edge_module, node_module, global_module)

    def forward(self, edge_features: list[Tensor], edge_indices: list[Tensor], edge_types: list[Tuple[str, str, str]],
                node_features: list[Tensor], node_types: list[str],
                global_features: Tensor, batches: list[Tensor]) -> Tuple[list[Tensor], list[Tensor], Tensor]:
        """
        Computes the forward pass for this message passing block

        Args:
            edge_features: storing edge_features for each edge_type
            edge_indices: storing edge_index for each edge_type
            edge_types: 3 strings that define a given edge: (src_node_type, edge_type, dest_node_type)
            node_features: storing node_features for each node_type
            node_types: string that define a given node_type
            global_features: Features for the whole graph. Has shape (num_global_features, num_graphs_in_batch)
            batches: Indexing for different graphs in the same badge. for each node_type

        Returns:
            Updated node, edge and global features (new_node_features, new_edge_features, new_global_features)
                for each type as a tuple
        """
        updated_edge_features, updated_node_features, updated_global_features\
            = self._meta_layer(edge_features, edge_indices, edge_types,
                               node_features, node_types, global_features, batches)

        if self._use_residual_connections:  # simply add original features to updated ones
            for i in range(len(edge_types)):
                updated_edge_features[i] = updated_edge_features[i] + edge_features[i]
            for i in range(len(node_types)):
                updated_node_features[i] = updated_node_features[i] + node_features[i]
            if global_features is not None:
                updated_global_features = updated_global_features + global_features

        return updated_edge_features, updated_node_features, updated_global_features
