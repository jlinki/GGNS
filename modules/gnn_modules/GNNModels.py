from typing import Dict
from torch import nn as nn

from modules.gnn_modules.heterogeneous_modules.HeteroGNNBase import HeteroGNNBase
from modules.gnn_modules.homogeneous_modules.GNNBase import GNNBase
from util.Types import ConfigDict


class HomoGNN(nn.Module):
    """
        Homogeneous Graph Neural Network (GNN) module to process the common graph including the mesh and point cloud.
        It uses the encoder and message passing stack of the GNNBase with a node-wise decoder on top.
    """
    def __init__(self,
                 in_node_features: Dict[str, int],
                 in_edge_features: Dict[str, int],
                 in_global_features: int,
                 out_node_features: int,
                 network_config: ConfigDict):
        """
        Args:
            in_node_features: Dictionary, where key is the node_type and values is the number of input node features in type
            in_edge_features: Dictionary, where key is the edge_types and values is the number of input edge features in type
            in_global_features: Number of input global features per graph
            out_node_features: Number of output node features (2 for 2D positions, 3 for 3D)
            network_config: Config containing information on how to build and train the overall network. Includes a config for the GNNBase.
              latent_dimension: how large the latent-dimension of the embedding should be
        """
        super().__init__()

        latent_dimension = network_config.get("latent_dimension")
        self.mlp_decoder = network_config.get("mlp_decoder")

        self._gnn_base = GNNBase(in_node_features=in_node_features,
                                 in_edge_features=in_edge_features,
                                 in_global_features=in_global_features,
                                 network_config=network_config)
        # define decoder
        if self.mlp_decoder:
            self.node_decoder1 = nn.Linear(latent_dimension, latent_dimension)
            self.node_decoder2 = nn.Linear(latent_dimension, out_node_features)
            self.activation = nn.LeakyReLU()
        else:
            self.node_decoder = nn.Linear(latent_dimension, out_node_features)

    def forward(self, tensor):
        """
        Performs a forward pass through the Full Graph Neural Network for the given input batch of homogeneous graphs
        Args:
            tensor: Batch of Data objects of pytorch geometric. Represents a number of homogeneous graphs
        Returns:
            Tuple.
            node_features: Decoded features of the nodse
            edge_features: Latent edges features of the last MP-Block
            global_features: Last latent global feature
        """
        node_features, edge_features, global_features = self._gnn_base(tensor)
        if self.mlp_decoder:
            node_features = self.node_decoder1(node_features)
            node_features = self.activation(node_features)
            node_features = self.node_decoder2(node_features)
        else:
            node_features = self.node_decoder(node_features)

        return node_features, edge_features, global_features


class HeteroGNN(nn.Module):
    """
    Heterogeneou Graph Neural Network (GNN) module to process the common graph including the mesh and point cloud.
    It uses the encoder and message passing stack of the HeteroGNNBase with a node-wise decoder for the 'mesh' nodes on top.
    """
    def __init__(self,
                 in_node_features: Dict[str, int],
                 in_edge_features: Dict[str, int],
                 in_global_features: int,
                 out_node_features: int,
                 network_config: ConfigDict):
        """
        Args:
            in_node_features: Dictionary, where keys are node_types and values are number of input node features in type
            in_edge_features: Dictionary, where keys are edge_types and values are number of input edge features in type
            in_global_features: Number of input global features per graph
            out_node_features: Number of output node features for node type 'mesh' (2 for 2D positions, 3 for 3D)
            network_config: Config containing information on how to build and train the overall network. 
              latent_dimension: how large the latent-dimension of the embedding should be
        """
        super().__init__()

        latent_dimension = network_config.get("latent_dimension")
        self.mlp_decoder = network_config.get("mlp_decoder")

        self._hetero_gnn_base = HeteroGNNBase(in_node_features=in_node_features,
                                              in_edge_features=in_edge_features,
                                              in_global_features=in_global_features,
                                              network_config=network_config)
        # define decoder
        if self.mlp_decoder:
            self.node_decoder1 = nn.Linear(latent_dimension, latent_dimension)
            self.node_decoder2 = nn.Linear(latent_dimension, out_node_features)
            self.activation = nn.LeakyReLU()
        else:
            self.node_decoder = nn.Linear(latent_dimension, out_node_features)

    def forward(self, tensor):
        """
        Performs a forward pass through the Full Graph Neural Network for the given input batch of heterogeneous graphs
        Args:
            tensor: Batch of Data objects of pytorch geometric. Represents a number of heterogeneous graphs
        Returns:
            Tuple.
            node_features: Decoded features of the node type 'mesh'
            edge_features_dict: Dictionary of last latent edges features of the last MP-Block
            global_features: Last latent global feature
        """
        node_features_dict, edge_features_dict, global_features, batch = self._hetero_gnn_base(tensor)
        if self.mlp_decoder:
            node_features = self.node_decoder1(node_features_dict['mesh'])
            node_features = self.activation(node_features)
            node_features = self.node_decoder2(node_features)
        else:
            node_features = self.node_decoder(node_features_dict['mesh'])

        return node_features, edge_features_dict, global_features
