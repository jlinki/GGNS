import torch

from modules.gnn_modules.AbstractMetaModule import AbstractMetaModule
from util.Types import *
from modules.MLP import MLP


class HomogeneousMetaModule(AbstractMetaModule):
    """
    Base class for the homogeneous modules used in the GNN.
    They are used for updating node-, edge-, and global features.
    """
    def __init__(self, in_features: int, num_types: int, latent_dimension: int, base_config: ConfigDict,
                 out_features: Optional[int] = None, aggregation_function_str: str = "mean"):
        """
        Args:
            in_features: The input shape for the feedforward network
            latent_dimension: Dimensionality of the internal layers of the mlp
            out_features: The output dimension for the feedforward network
            base_config: Dictionary specifying the way that the gnn base should look like
            aggregation_function_str: How to aggregate over the nodes/edges/globals. Defaults to "mean" aggregation,
              which corresponds to torch_scatter.scatter_mean()
        """
        super().__init__(aggregation_function_str)
        mlp_config = base_config.get("mlp")
        self._out_mlp = MLP(in_features=in_features, config=mlp_config, latent_dimension=latent_dimension,
                            out_features=out_features)

    @property
    def out_features(self) -> int:
        """
        Size of the features the forward function returns.
        """
        return self._out_mlp.out_features


class EdgeModule(HomogeneousMetaModule):
    def forward(self, src: torch.Tensor, dest: torch.Tensor,
                edge_attr: torch.Tensor, u: Optional[torch.Tensor], batch: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute edge updates for the edges of the Module
        Args:
            src: (num_edges, num_node_features)
              Represents the source nodes of the graph(s), i.e., a node for each outgoing edge.
            dest: (num_edges, num_node_features)
              Represents the target nodes of the graph(s), i.e., a node for each incoming edge.
            edge_attr: (num_edges, num_edge_features). The features on each edge
            u: A matrix (num_graphs_in_batch, global_feature_dimension). Distributed along the graphs
            batch: A tensor [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1] that assigns a graph to each node

        Returns: An updated representation of the edge attributes

        """
        if u is None:  # no global information
            aggregated_features = torch.cat([src, dest, edge_attr], 1)
        else:  # has global information
            aggregated_features = torch.cat([src, dest, edge_attr, u[batch]], 1)

        out = self._out_mlp(aggregated_features)
        return out


class NodeModule(HomogeneousMetaModule):
    def forward(self, x, edge_index, edge_attr, u, batch):
        """
            Compute updates for each node feature vector as x_i' = f2(x_i, agg_j f1(e_ij, x_j), u),
            where f1 and f2 are MLPs
            Args:
                x: (num_nodes, num_node_features). Feature matrix for all nodes of the graphs
                edge_index: (2, num_edges). Sparse representation of the source and target nodes of each edge.
                edge_attr: (num_edges, num_edge_features). The features on each edge
                u: A matrix (num_graphs_in_batch, global_feature_dimension). Distributed along the graphs
                batch: A tensor [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1] that assigns a graph to each node
            Returns: An updated representation of the global features for each graph
        """
        src, des = edge_index  # split edges in source and target nodes
        # get source node features of all edges, combine with edge feature

        aggregated_neighbor_features = self._aggregation_function(edge_attr, src, dim=0, dim_size=x.size(0))
        if u is None:  # no global information
            aggregated_features = torch.cat([x, aggregated_neighbor_features], dim=1)
        else:  # has global information
            aggregated_features = torch.cat([x, aggregated_neighbor_features, u[batch]], dim=1)

        out = self._out_mlp(aggregated_features)
        return out


class GlobalModule(HomogeneousMetaModule):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                u: Optional[torch.Tensor], batch: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute updates for the global feature vector of each graph.
        u' = mlp(u, agg e, agg v)
        Args:
            x: (num_nodes, num_node_features). Feature matrix for all nodes of the graphs
            edge_index: (2, num_edges). Sparse representation of the source and target nodes of each edge.
            edge_attr: (num_edges, num_edge_features). The features on each edge
            u: A matrix (num_graphs_in_batch, global_feature_dimension). Distributed along the graphs
            batch: A tensor [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1] that assigns a graph to each node

        Returns: An updated representation of the global features for each graph. Has shape (num_graphs, latent_dim)
        """

        edge_batch = batch[edge_index[0]]
        # e.g., edge_index[0] = [0,1,1,2,1,1,0,1,1,0,||| 3,4,3,4,3,4,||| 6,5,7,5,8] -->
        # batch[edge_index[0] = [0,0,0,0,0,0,0,0,0,0,||| 1,1,1,1,1,1,||| 2,2,2,2,2]

        graphwise_node_aggregation = self._aggregation_function(x, batch, dim=0)
        graphwise_edge_aggregation = self._aggregation_function(edge_attr, edge_batch, dim=0)

        if u is None:
            aggregated_features = torch.cat([graphwise_node_aggregation, graphwise_edge_aggregation], dim=1)
        else:
            aggregated_features = torch.cat([u, graphwise_node_aggregation, graphwise_edge_aggregation], dim=1)

        out = self._out_mlp(aggregated_features)
        return out

class GlobalModuleNoUpdate(HomogeneousMetaModule):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                u: Optional[torch.Tensor], batch: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Returns input global features u without update

        """
        return u
