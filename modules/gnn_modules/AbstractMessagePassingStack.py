import torch.nn as nn

from modules.gnn_modules.AbstractMessagePassingBlock import AbstractMessagePassingBlock
from util.Types import *


class AbstractMessagePassingStack(nn.Module):
    """
    Message Passing module that acts on both node and edge features used for observation graphs.
    Internally stacks multiple instances of MessagePassingBlock.
    """
    def __init__(self, base_config: ConfigDict):
        """
        Args:
            base_config: Dictionary specifying the way that the gnn base should look like.
                num_blocks: how many blocks this stack should have
                use_residual_connections: if the blocks should use residual connections
        """
        super().__init__()
        num_blocks: int = base_config.get("num_blocks")
        use_residual_connections: bool = base_config.get("use_residual_connections")

        self._blocks = nn.ModuleList([
            AbstractMessagePassingBlock(base_config=base_config, use_residual_connections=use_residual_connections)
            for _ in range(num_blocks)])

    @property
    def num_blocks(self) -> int:
        """
        How many blocks this stack is composed of.
        """
        return len(self._blocks)

    @property
    def out_node_features(self) -> int:
        """
        The node feature dimension that the last block in the stack returns.
        """
        return self._blocks[-1].out_node_features

    @property
    def out_edge_features(self) -> int:
        """
        The edge feature dimension that the last block in the stack returns.
        """
        return self._blocks[-1].out_edge_features

    @property
    def out_global_features(self) -> int:
        """
        The global feature dimension that the last block in the stack returns.
        """
        return self._blocks[-1].out_global_features
