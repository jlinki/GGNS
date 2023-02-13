import torch

from util.Functions import get_aggregation_function


class AbstractMetaModule(torch.nn.Module):
    """
    An abstract class for the modules used in the GNN. They are used for updating node-, edge-, and global features.
    """
    def __init__(self, aggregation_function_str: str = "mean"):
        """
        Args:
            aggregation_function_str: How to aggregate over the nodes/edges/globals. Defaults to "mean" aggregation,
            which corresponds to torch_scatter.scatter_mean()
        """
        super().__init__()

        self._aggregation_function_str = aggregation_function_str
        self._aggregation_function = get_aggregation_function(aggregation_function_str)

    def extra_repr(self) -> str:
        """
        To be printed when using print(self) on either this class or some module implementing it
        Returns: A string containing information about the parameters of this module
        """
        return f"aggregation function: {self._aggregation_function_str}"

    @property
    def out_features(self) -> int:
        """
        Size of the features the forward function returns.
        """
        raise NotImplementedError("AbstractMetaModule does not implement num_global_features")
