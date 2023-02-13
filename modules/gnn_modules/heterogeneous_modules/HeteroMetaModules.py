import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(os.getcwd()).parent.parent))  # path hacking

import torch
import torch.nn as nn

from util.Types import *
import util.Keys as Keys
from modules.gnn_modules.homogeneous_modules.MetaModules import AbstractMetaModule
from modules.MLP import MLP


class HeterogeneousMetaModule(AbstractMetaModule):
    """
    Base class for the heterogeneous modules used in the GNN.
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
        self.het_neighbor_aggregation = base_config.get("het_neighbor_aggregation")
        self.het_edge_shared_weights = bool(base_config.get("het_edge_shared_weights"))
        self.het_node_shared_weights = bool(base_config.get("het_node_shared_weights"))
        mlp_config = base_config.get("mlp")
        if self.het_edge_shared_weights or self.het_node_shared_weights:
            self._mlp = MLP(in_features=in_features, config=mlp_config,
                        latent_dimension=latent_dimension, out_features=out_features)
        elif (not self.het_edge_shared_weights) or (not self.het_node_shared_weights):
            self._mlp_list = nn.ModuleList([
                MLP(in_features=in_features, config=mlp_config,
                    latent_dimension=latent_dimension, out_features=out_features)
                for _ in range(num_types)])

    @property
    def out_features(self) -> int:
        """
        Size of the features the forward function returns.
        """
        if self.het_edge_shared_weights or self.het_node_shared_weights:
            return self._mlp.out_features
        else:
            return self._mlp_list[0].out_features


class HeteroEdgeModule(AbstractMetaModule):
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
        self.het_neighbor_aggregation = base_config.get("het_neighbor_aggregation")
        self.het_edge_shared_weights = bool(base_config.get("het_edge_shared_weights"))
        mlp_config = base_config.get("mlp")
        if self.het_edge_shared_weights:
            self._mlp = MLP(in_features=in_features, config=mlp_config,
                        latent_dimension=latent_dimension, out_features=out_features)
        else:
            self._mlp_list = nn.ModuleList([
                MLP(in_features=in_features, config=mlp_config,
                    latent_dimension=latent_dimension, out_features=out_features)
                for _ in range(num_types)])

    @property
    def out_features(self) -> int:
        """
        Size of the features the forward function returns.
        """
        if self.het_edge_shared_weights:
            return self._mlp.out_features
        else:
            return self._mlp_list[0].out_features

    def forward(self, edge_features: list[Tensor], edge_indices: list[Tensor], edge_types: list[Tuple[str, str, str]],
                node_features: list[Tensor], node_types: list[str],
                global_features: Tensor, batches: list[Tensor]) -> list[Tensor]:
        """
        Compute edge updates for the edges of the Module for heterogeneous graphs
        Args:
            edge_features: storing edge_features for each edge_type
            edge_indices: storing edge_index for each edge_type
            edge_types: 3 strings that define a given edge: (src_node_type, edge_type, dest_node_type)
            node_features: storing node_features for each node_type
            node_types: string that define a given node_type
            global_features: Features for the whole graph. Has shape (num_global_features, num_graphs_in_batch)
            batches: list of tensors [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1]
              that assigns a graph to each node for each node_type

        Returns: An updated representation of the edge attributes for all edge_types
        """
        out: list[Tensor] = []
        for edge_type_i in range(len(edge_types)):
            edge_attr = edge_features[edge_type_i]

            # calculate source node features and dest node features for each edge in this type
            source_node_type, _, dest_node_type = edge_types[edge_type_i]
            found_type_index = 0
            edge_source_nodes = Tensor()
            edge_dest_nodes = Tensor()
            for node_type_i in range(len(node_types)):
                curr_node_type = node_types[node_type_i]
                source_indices, dest_indices = edge_indices[edge_type_i]
                nodes = node_features[node_type_i]
                found_type_index = curr_node_type
                if curr_node_type == source_node_type:
                    edge_source_nodes = nodes[source_indices]
                if curr_node_type == dest_node_type:
                    edge_dest_nodes = nodes[dest_indices]

            # concatenate everything
            if global_features is None:  # no global information
                aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, edge_attr], 1)
            else:  # has global information
                aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, edge_attr,
                                                 global_features[batches[found_type_index]]], 1)

            # update
            if self.het_edge_shared_weights:
                updated_edge_attr = self._mlp(aggregated_features)
            else:
                updated_edge_attr = self._mlp_list[edge_type_i](aggregated_features)
            out.append(updated_edge_attr)
        return out


class HeteroNodeModule(AbstractMetaModule):
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
        self.het_neighbor_aggregation = base_config.get("het_neighbor_aggregation")
        self.het_node_shared_weights = bool(base_config.get("het_node_shared_weights"))
        mlp_config = base_config.get("mlp")
        if self.het_node_shared_weights:
            self._mlp = MLP(in_features=in_features, config=mlp_config,
                        latent_dimension=latent_dimension, out_features=out_features)
        else:
            self._mlp_list = nn.ModuleList([
                MLP(in_features=in_features, config=mlp_config,
                    latent_dimension=latent_dimension, out_features=out_features)
                for _ in range(num_types)])

    @property
    def out_features(self) -> int:
        """
        Size of the features the forward function returns.
        """
        if self.het_node_shared_weights:
            return self._mlp.out_features
        else:
            return self._mlp_list[0].out_features

    def forward(self, edge_features: list[Tensor], edge_indices: list[Tensor], edge_types: list[Tuple[str, str, str]],
                node_features: list[Tensor], node_types: list[str],
                global_features: Tensor, batches: list[Tensor]) -> list[Tensor]:
        """
        Compute updates for each node feature vector as x_i' = f2(x_i, agg_j f1(e_ij, x_j), u),
        where f1 and f2 are MLPs
        Args:
            edge_features: storing edge_features for each edge_type
            edge_indices: storing edge_index for each edge_type
            edge_types: 3 strings that define a given edge: (src_node_type, edge_type, dest_node_type)
            node_features: storing node_features for each node_type
            node_types: string that define a given node_type
            global_features: Features for the whole graph. Has shape (num_global_features, num_graphs_in_batch)
            batches: list of tensors [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1]
              that assigns a graph to each node for each node_type

        Returns: An updated representation of the node features for each node_type for the graph
        """
        device = edge_features[0].device
        out: list[Tensor] = []
        for node_type_i in range(len(node_types)):
            curr_node_features: Tensor = node_features[node_type_i]

            # concatenate each edge-type that has this node-type as a source and create indices for aggregating them
            src_concat_edge_features = Tensor().to(device)
            src_indices = Tensor().to(device)
            for edge_type_i in range(len(edge_types)):
                source_node_type, _, dest_node_type = edge_types[edge_type_i]
                if dest_node_type == node_types[node_type_i]:
                    curr_edge_features = edge_features[edge_type_i]
                    source_indices, dest_indices = edge_indices[edge_type_i]
                    aggregated_edge_type = self._aggregation_function(curr_edge_features, dest_indices, dim=0, dim_size=curr_node_features.size(0))
                    src_concat_edge_features = torch.cat([src_concat_edge_features, aggregated_edge_type], dim=1)
                    edge_indices_vector = torch.arange(start=0, end=curr_node_features.shape[1])[:, None].t().to(device)
                    src_indices = torch.cat([src_indices, edge_indices_vector], dim=1).to(torch.int64)

            if src_concat_edge_features.shape[0] != 0:
                # on aggr(aggr()) we want to aggregate all the edge-features we have concatenated.
                if self.het_neighbor_aggregation == Keys.AGGR_AGGR:
                    src_concat_edge_features = \
                        self._aggregation_function(src_concat_edge_features, src_indices, dim=1)

                # concatenate everything
                if global_features is None:  # no global information
                    aggregated_features = torch.cat([curr_node_features, src_concat_edge_features], dim=1)
                else:  # has global information
                    aggregated_features = torch.cat([curr_node_features, src_concat_edge_features,
                                                     global_features[batches[node_type_i]]], dim=1)

                # update
                if self.het_node_shared_weights:
                    updated_node_feature = self._mlp(aggregated_features)
                else:
                    updated_node_feature = self._mlp_list[node_type_i](aggregated_features)
            else:
                updated_node_feature = curr_node_features

            out.append(updated_node_feature)

        return out


# todo fix this!
class HeteroGlobalModule(HeterogeneousMetaModule):
    def forward(self, edge_features: list[Tensor], edge_indices: list[Tensor], edge_types: list[Tuple[str, str, str]],
                node_features: list[Tensor], node_types: list[str],
                global_features: Tensor, batches: list[Tensor]) -> torch.Tensor:
        """
        Compute updates for the global feature vector of each graph.
        u' = mlp(u, agg e, agg v)
        Args:
            edge_features: storing edge_features for each edge_type
            edge_indices: storing edge_index for each edge_type
            edge_types: 3 strings that define a given edge: (src_node_type, edge_type, dest_node_type)
            node_features: storing node_features for each node_type
            node_types: string that define a given node_type
            global_features: Features for the whole graph. Has shape (num_global_features, num_graphs_in_batch)
            batches: list of tensors [0, 0, ..., 0, 1, ..., 1, ..., num_graphs-1]
              that assigns a graph to each node for each node_type

        Returns: An updated representation of the global features for each graph. Has shape (num_graphs, latent_dim)
        """
        batch = batches[0]
        # node aggregation
        aggregate_node_features = Tensor
        for node_type_i in range(len(node_types)):
            aggregate_node_type = self._aggregation_function(node_features[node_type_i], batch, dim=0)
            torch.cat([aggregate_node_features, aggregate_node_type], dim=1)
        if self.het_neighbor_aggregation == Keys.AGGR_AGGR:
            aggregate_node_features = self._aggregation_function(aggregate_node_features, batch.t(), dim=1)

        # edge aggregation
        aggregate_edge_features = Tensor
        for edge_type_i in range(len(edge_types)):
            edge_batch = batch[edge_indices[edge_type_i][0]]
            # e.g., edge_index[0] = [0,1,1,2,1,1,0,1,1,0,||| 3,4,3,4,3,4,||| 6,5,7,5,8] -->
            # batch[edge_index[0] = [0,0,0,0,0,0,0,0,0,0,||| 1,1,1,1,1,1,||| 2,2,2,2,2]

            aggregate_edge_type = self._aggregation_function(edge_features[edge_type_i], edge_batch, dim=0)
            torch.cat([aggregate_edge_features, aggregate_edge_type], dim=1)
        # todo How can I calc the edge_batch for all edge_types?
        complete_edge_batch = torch.zeros((0, 0))
        if self.het_neighbor_aggregation == Keys.AGGR_AGGR:
            aggregate_edge_features = \
                self._aggregation_function(aggregate_edge_features, complete_edge_batch.t(), dim=1)

        if global_features is None:
            aggregated_features = torch.cat([aggregate_node_features, aggregate_edge_features], dim=1)
        else:
            aggregated_features = torch.cat([aggregate_node_features, aggregate_edge_features,
                                             global_features], dim=1)

        out = self.out_mlp(aggregated_features)
        return out
