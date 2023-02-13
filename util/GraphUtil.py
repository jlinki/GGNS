from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx


def draw_graph(graph: Data) -> None:
    """
    Utility function to draw a given graph object. Draws nodes and edges of the graph
    Args:
        graph: Pytorch Geometric Data object storing a single graph
    # todo: adapt to plotly [https://plotly.com/python/network-graphs/]

    Returns: Nothing, but plots the graph in the current matplotlib figure

    """
    networkx_graph = to_networkx(graph)
    if graph.pos is not None and graph.pos[0].shape == (2,):  # 2-dimensional positions
        positions = {idx: node_position.numpy() for idx, node_position in enumerate(graph.pos)}
    else:
        positions = None
    nx.draw_networkx(networkx_graph, pos=positions)
