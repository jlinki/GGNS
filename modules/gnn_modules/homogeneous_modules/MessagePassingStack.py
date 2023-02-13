import torch
from torch import nn as nn

from modules.gnn_modules.AbstractMessagePassingStack import AbstractMessagePassingStack
from util.Types import *
from modules.gnn_modules.homogeneous_modules.MessagePassingBlock import MessagePassingBlock


class MessagePassingStack(AbstractMessagePassingStack):
    """
    Message Passing module that acts on both node and edge features.
    Internally stacks multiple instances of MessagePassingBlocks.
    This implementation is used for homogeneous observation graphs.
    """
    def __init__(self, base_config: ConfigDict, latent_dimension: int, use_global_features: bool,
                 aggregation_function_str: str):
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
        self._blocks = nn.ModuleList([MessagePassingBlock(base_config=base_config, in_node_features=latent_dimension,
                                                          in_edge_features=latent_dimension,
                                                          in_global_features=in_global_features,
                                                          aggregation_function_str=aggregation_function_str,
                                                          use_residual_connections=use_residual_connections)
                                      for _ in range(num_blocks)])

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor,
                global_features: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass for this homogeneous message passing stack.

        Args:
            node_features: The features for each node of the graph. Has shape (num_nodes, num_features_per_node)
            edge_index: Connectivity Tensor of the graph. Has shape (2, num_edges)
            edge_features: Feature matrix of the edges. Has shape (num_edges, num_features_per_edge)
            global_features: Features for the whole graph. Has shape (num_global_features, num_graphs_in_batch)
            batch: Indexing for different graphs in the same badge.

        Returns:
            Updated node, edge and global features is a tuple

        """
        for message_passing_block in self._blocks:
            node_features, edge_features, global_features = \
                message_passing_block(node_features=node_features,
                                      edge_index=edge_index, edge_features=edge_features,
                                      global_features=global_features, batch=batch)

        output_tensors = node_features, edge_features, global_features
        return output_tensors
