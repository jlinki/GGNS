from typing import Tuple
import torch
from torch import Tensor


class HeteroMetaLayer(torch.nn.Module):
    """
    This MetaLayer combines the three Modules (Edge, Node, Global) into a single forward pass.
    It is only used for heterogeneous observation graphs.
    """

    def __init__(self, edge_model=None, node_model=None, global_model=None):
        """
        Args:
            edge_model: The edge module used, which updates the edge features.
            node_model: The node module used, which updates the node features.
            global_model: The global module used, which updates the global features.
        """
        super(HeteroMetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        """
        This resets all the parameters for all modules
        """
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, edge_features: list[Tensor], edge_indices: list[Tensor], edge_types: list[Tuple[str, str, str]],
                node_features: list[Tensor], node_types: list[str],
                global_features: Tensor, batches: list[Tensor]) -> Tuple[list[Tensor], list[Tensor], Tensor]:
        """
        Computes the forward pass for this heterogeneous meta layer

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
        if self.edge_model is not None:
            edge_features = \
                self.edge_model(edge_features, edge_indices, edge_types,
                                node_features, node_types, global_features, batches)

        if self.node_model is not None:
            node_features = \
                self.node_model(edge_features, edge_indices, edge_types,
                                node_features, node_types, global_features, batches)

        if self.global_model is not None:
            global_features = \
                self.global_model(edge_features, edge_indices, edge_types,
                                  node_features, node_types, global_features, batches)

        return edge_features, node_features, global_features

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)
