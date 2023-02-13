import copy
import torch
import torch.nn as nn

from modules.gnn_modules.AbstractGNNBase import AbstractGNNBase
from util.Types import *
from modules.gnn_modules.heterogeneous_modules.HeteroMessagePassingStack \
    import HeteroMessagePassingStack
from modules.HelperModules import LinearEmbedding
from modules.gnn_modules.homogeneous_modules.GNNBase import get_global_features


class HeteroGNNBase(AbstractGNNBase):
    """
        Graph Neural Network (GNN) Base module processes the graph observations of the environment.
        It uses a stack of multiple GNN Blocks. Each block defines a single GNN pass.
        The forward functions is defined by implementations of this abstract class.
        This implementation is used for heterogeneous observation graphs.
    """
    def __init__(self, in_node_features: Dict[str, int], in_edge_features: Dict[str, int], in_global_features: int,
                 network_config: ConfigDict):
        """
        Args:
            in_node_features: Dictionary, where keys are node_types and values are number of input node features in type
            in_edge_features: Dictionary, where keys are edge_types and values are number of input edge features in type
            in_global_features: Number of input global features per graph
            network_config: Config containing information on how to build and train the overall network. Includes
              a config for this base.
              latent_dimension: how large the latent-dimension of the embedding should be
              base: more config used for the message passing stack
        """
        super().__init__(network_config)

        base_config = network_config.get("base")
        latent_dimension = network_config.get("latent_dimension")

        # check if input size of 0 works
        in_edge_features = in_edge_features if in_edge_features is not None else {"": 0}
        in_global_features = 0 if (not self._use_global_features) or in_global_features is None else in_global_features

        # create embeddings
        self._edge_input_embeddings = nn.ModuleList([
            LinearEmbedding(in_features=num_edge_features, out_features=latent_dimension)
            for _, num_edge_features in in_edge_features.items()])

        self._node_input_embeddings = nn.ModuleList([
            LinearEmbedding(in_features=num_node_features, out_features=latent_dimension)
            for _, num_node_features in in_node_features.items()])

        if self._use_global_features:
            self._global_input_embedding = LinearEmbedding(in_features=in_global_features,
                                                           out_features=latent_dimension)

        # create message passing stack
        num_edge_types = len(in_edge_features.keys())
        num_node_types = len(in_node_features.keys())
        self._message_passing_stack = HeteroMessagePassingStack(base_config=base_config,
                                                                latent_dimension=latent_dimension,
                                                                use_global_features=self._use_global_features,
                                                                aggregation_function_str=self._aggregation_function_str,
                                                                num_edge_types=num_edge_types,
                                                                num_node_types=num_node_types)

    def _unpack_features(self, tensor: InputBatch):
        """
        Unpacking important data from heterogeneous graphs.
        Args:
            tensor (): The input heterogeneous observation

        Returns:
            Tuple of edge_features, edge_index, node_features, global_features and batch
        """
        # edge features
        edge_features = []
        edge_indices = []
        for i in range(len(tensor.edge_stores)):
            edge_features.append(copy.deepcopy(tensor.edge_stores[i].edge_attr))
            edge_indices.append(tensor.edge_stores[i].edge_index)

        # node features
        node_features = []
        for i in range(len(tensor.node_stores)):
            node_features.append(copy.deepcopy(tensor.node_stores[i].x))

        # global features
        global_features = get_global_features(graph_tensor=tensor) if self._use_global_features else None
        batches = []
        for i in range(len(tensor.node_stores)):
            if hasattr(tensor.node_stores[0], "batch"):
                batches.append(tensor.node_stores[i].batch)
            else:
                batches.append(torch.zeros(tensor.node_stores[i].num_nodes).long())

        return edge_features, edge_indices, node_features, global_features, batches

    def forward(self, tensor: InputBatch) \
            -> Tuple[Dict, Dict, Optional[torch.Tensor], Optional[Dict]]:
        """
        Performs a forward pass through the Full Graph Neural Network for the given input batch of heterogeneous graphs

        Args:
            tensor: Batch of Data objects of pytorch geometric. Represents a number of heterogeneous graphs
        Returns:
            Tuple.
                First entry is a dictionary with the updated node-types as a key and node-features as a feature.
                Second entry is a dictionary with the updated edge-types as a key and edge-features and edge-types
                as a feature.
                Third entry is the updated global feature.
                Fourth entry is the input batches.
        """
        if isinstance(tensor, Data):
            raise ValueError(f"HeteroGNNBase was called using a Homogenous Graph, this should not happen!")
        assert(len(tensor.node_types) == len(self._node_input_embeddings) and
               len(tensor.edge_types) == len(self._edge_input_embeddings))

        # unpack data from input batch
        edge_types = tensor.edge_types
        node_types = tensor.node_types
        edge_features, edge_indices, node_features, global_features, batches = self._unpack_features(tensor)

        # get input embeddings of nodes, edges and globals
        for i in range(len(edge_types)):
            edge_features[i] = self._edge_input_embeddings[i](edge_features[i])
        for i in range(len(node_types)):
            node_features[i] = self._node_input_embeddings[i](node_features[i])
        if self._use_global_features:
            global_features = self._global_input_embedding(global_features)

        # do message passing
        edge_features, node_features, global_features = \
            self._message_passing_stack(edge_features=edge_features,
                                        edge_indices=edge_indices,
                                        edge_types=edge_types,
                                        node_features=node_features,
                                        node_types=node_types,
                                        global_features=global_features,
                                        batches=batches)

        # package updated features
        edge_index_dict = dict(zip(edge_types, edge_indices))
        edge_feature_dict = dict(zip(edge_types, edge_features))
        edge_dict = {}
        for key, value in edge_index_dict.items():
            edge_dict[key] = {"edge_index": edge_index_dict[key], "edge_attr": edge_feature_dict[key]}
        return dict(zip(node_types, node_features)), edge_dict, global_features, dict(zip(node_types, batches))
