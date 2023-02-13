import torch
import torch.nn as nn

from modules.gnn_modules.AbstractMessagePassingStack import \
    AbstractMessagePassingStack
from util.Types import *


def get_global_features(graph_tensor: Union[Data, HeteroData]) -> torch.Tensor:
    """
    Unpacks the global features of Data or HeteroData
    Args:
        graph_tensor: The tensor to unpack global features from
    Returns:
        Empty tensor if no global features could be found, otherwise the global features
    """
    global_features = graph_tensor.u if hasattr(graph_tensor, "u") else torch.tensor([])
    batch = graph_tensor.batch if hasattr(graph_tensor, "batch") else None

    if batch is None:  # only one graph
        if global_features.dim() == 0:
            global_features = global_features.view(1,1)
        if global_features.dim() == 1:
            global_features = global_features[None, :]
        else:
            global_features = global_features.view(1, -1)
    else:  # Reshape global features to fit the graph
        assert hasattr(graph_tensor, "ptr"), "Need pointer for graph ids when using batch and global features"
        num_graphs = len(graph_tensor.ptr) - 1
        if len(global_features > 0):  # Reshape global features to fit the graph
            global_features = global_features.reshape((-1, int(len(global_features) / num_graphs)))
        else:  # No global features. make a bigger placeholder
            global_features = global_features[None, :][[0] * num_graphs]
    global_features = global_features.float()

    return global_features


class AbstractGNNBase(nn.Module):
    """
        Graph Neural Network (GNN) Base module processes the graph observations of the environment.
        It uses a stack of multiple GNN Blocks. Each block defines a single GNN pass.
        The forward functions is defined by implementations of this abstract class.
    """
    def __init__(self, network_config: ConfigDict):
        """
        Args:
            network_config: Config containing information on how to build and train the overall network.
              Includes a config for this base.
              aggregation_function: which aggregation function use used for
              use_global_features: whether to use global features.
        """
        super().__init__()

        base_config = network_config.get("base")
        self._aggregation_function_str = network_config.get("aggregation_function")
        self._use_global_features: bool = base_config.get("use_global_features")

        # create message passing stack
        self._message_passing_stack: AbstractMessagePassingStack = \
            AbstractMessagePassingStack(base_config=base_config)

    @property
    def out_node_features(self) -> int:
        """
        Query how many node features the GNN Base will return.
        Returns:
            number of node features.
        """
        if self._message_passing_stack.num_blocks > 0:  # has message passing block
            return self._message_passing_stack.out_node_features
        else:  # is just linear embedding
            return self._node_input_embeddings[-1].out_features

    @property
    def out_edge_features(self) -> int:
        """
        Query how many edge features the GNN Base will return.
        Returns:
            number of edge features.
        """
        if self._message_passing_stack.num_blocks > 0:
            return self._message_passing_stack.out_edge_features
        else:
            return self._edge_input_embeddings[-1].out_features

    @property
    def out_global_features(self) -> int:
        """
        Query how many edge features the GNN Base will return.
        Returns:
            number of edge features.
        """
        if self._message_passing_stack.num_blocks > 0:
            return self._message_passing_stack.out_global_features
        elif self._global_input_embedding is not None:
            return self._global_input_embedding.out_features
        else:
            return 0

    @property
    def aggregation_function_str(self):
        """
        Query the aggregation function string
        Returns:
            aggregation function string
        """
        return self._aggregation_function_str
