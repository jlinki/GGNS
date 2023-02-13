from torch import nn
from modules.gnn_modules.AbstractMessagePassingStack import AbstractMessagePassingStack
from util.Types import *

from modules.gnn_modules.heterogeneous_modules.HeteroMessagePassingBlock \
    import HeteroMessagePassingBlock


class HeteroMessagePassingStack(AbstractMessagePassingStack):
    """
    Message Passing module that acts on both node and edge features.
    Internally stacks multiple instances of MessagePassingBlocks.
    This implementation is used for heterogeneous observation graphs.
    """
    def __init__(self, base_config: ConfigDict, latent_dimension: int, use_global_features: bool,
                 aggregation_function_str: str, num_edge_types: int, num_node_types: int):
        """
        Args:
            base_config: Dictionary specifying the way that the gnn base should look like.
                num_blocks: how many blocks this stack should have
                use_residual_connections: if the blocks should use residual connections. If True,
              the original inputs will be added to the outputs.
        """
        super().__init__(base_config)

        num_blocks: int = base_config.get("num_blocks")
        use_residual_connections: bool = base_config.get("use_residual_connections")

        in_global_features = latent_dimension if use_global_features else 0
        self._blocks = nn.ModuleList([
            HeteroMessagePassingBlock(base_config=base_config, in_node_features=latent_dimension,
                                      in_edge_features=latent_dimension, in_global_features=in_global_features,
                                      aggregation_function_str=aggregation_function_str,
                                      use_residual_connections=use_residual_connections,
                                      num_node_types=num_node_types, num_edge_types=num_edge_types)
            for _ in range(num_blocks)])

    def forward(self, edge_features: list[Tensor], edge_indices: list[Tensor], edge_types: list[Tuple[str, str, str]],
                node_features: list[Tensor], node_types: list[str],
                global_features: Tensor, batches: list[Tensor]) -> Tuple[list[Tensor], list[Tensor], Tensor]:
        """
        Computes the forward pass for this heterogeneous message passing stack.

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
        for message_passing_block in self._blocks:
            edge_features, node_features, global_features = \
                message_passing_block(edge_features=edge_features, edge_indices=edge_indices, edge_types=edge_types,
                                      node_features=node_features, node_types=node_types,
                                      global_features=global_features, batches=batches)

        return edge_features, node_features, global_features
