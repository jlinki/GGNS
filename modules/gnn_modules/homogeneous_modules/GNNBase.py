import torch

from modules.gnn_modules.AbstractGNNBase import get_global_features, AbstractGNNBase
from util.Types import *
import util.Keys as Keys
from modules.gnn_modules.homogeneous_modules.MessagePassingStack import MessagePassingStack
from modules.HelperModules import LinearEmbedding


class GNNBase(AbstractGNNBase):
    """
        Graph Neural Network (GNN) Base module processes the graph observations of the environment.
        It uses a stack of multiple GNN Blocks. Each block defines a single GNN pass.
        The forward functions is defined by implementations of this abstract class.
        This implementation is used for homogeneous observation graphs.
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

        latent_dimension = network_config.get("latent_dimension")
        base_config = network_config.get("base")

        # create embeddings
        num_edge_features = list(in_edge_features.values())[0]
        self._edge_input_embedding = LinearEmbedding(in_features=num_edge_features, out_features=latent_dimension)

        num_node_features = list(in_node_features.values())[0]
        self._node_input_embedding = LinearEmbedding(in_features=num_node_features, out_features=latent_dimension)

        if self._use_global_features:
            self._global_input_embedding = LinearEmbedding(in_features=in_global_features,
                                                           out_features=latent_dimension)

        # create message passing stack
        self._message_passing_stack = MessagePassingStack(base_config=base_config,
                                                          latent_dimension=latent_dimension,
                                                          use_global_features=self._use_global_features,
                                                          aggregation_function_str=self._aggregation_function_str)

    def _unpack_features(self, tensor: InputBatch):
        """
        Unpacking important data from homogeneous graphs.
        Args:
            tensor (): The input homogeneous observation

        Returns:
            Tuple of edge_features, edge_index, node_features, global_features and batch
        """
        # edge features
        edge_features = tensor.edge_attr
        edge_index = tensor.edge_index.long()  # cast to long for scatter operators

        # node features
        node_features = tensor.x if tensor.x is not None else tensor.pos

        # global features
        global_features = get_global_features(graph_tensor=tensor) if self._use_global_features else None
        batch = tensor.batch if hasattr(tensor, "batch") else None
        if batch is None:
            device = node_features.device
            batch = torch.zeros(node_features.shape[0]).long().to(device)

        return edge_features, edge_index, node_features, global_features, batch

    def forward(self, tensor: InputBatch) -> Tuple[Dict, Dict, Optional[torch.Tensor], Optional[Dict]]:
        """
        Performs a forward pass through the Full Graph Neural Network for the given input batch of homogeneous graphs

        Args:
            tensor: Batch of Data objects of pytorch geometric. Represents a number of homogeneous graphs
        Returns:
            Tuple.
                First entry is a dictionary with the updated single node-type as a key and node-feature as a feature.
                Second entry is a dictionary with the updated single edge-type as a key and edge-feature and edge-type
                as a feature.
                Third entry is the updated global feature.
                Fourth entry is the input batches.
        """
        # receive values from input batch
        if isinstance(tensor, HeteroData):
            raise ValueError(f"Normal GNNBase was called using a Heterogeneous Graph, this should not happen!")

        # unpack data from input batch
        edge_features, edge_index, node_features, global_features, batch = self._unpack_features(tensor)

        # get input embeddings of nodes, edges and globals
        node_features = self._node_input_embedding(node_features)
        edge_features = self._edge_input_embedding(edge_features)
        if self._use_global_features:
            global_features = self._global_input_embedding(global_features)

        # do the actual message passing GNN
        node_features, edge_features, global_features = self._message_passing_stack(node_features=node_features,
                                                                                    edge_index=edge_index,
                                                                                    edge_features=edge_features,
                                                                                    global_features=global_features,
                                                                                    batch=batch)

        # package updates as dictionaries
        return node_features, edge_features, global_features

            # {Keys.AGENT: node_features}, \
            #    {Keys.AGENT + "2" + Keys.AGENT: {"edge_index": edge_index, "edge_attr": edge_features}}, \
            # global_features, {Keys.AGENT: batch}
