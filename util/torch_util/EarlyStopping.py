from copy import deepcopy
from util.Types import *
from AbstractArchitecture import AbstractArchitecture
import numpy as np


class EarlyStopping:
    """
    Early Stopping utility for network training. Measures the network performance for some given metric and tells it to
    stop if it does not improve for a certain number of iterations.
    Can also store the currently best model state, which can be used to restore the best intermediate model
    after the training has finished
    """

    def __init__(self, early_stopping_config: ConfigDict, network: Optional[AbstractArchitecture] = None):
        patience = early_stopping_config.get("patience")
        restore_best = early_stopping_config.get("restore_best")
        warmup = early_stopping_config.get("warmup")
        self._max_warmup = warmup
        self._current_warmup = warmup
        self._patience = patience
        self._current_patience = patience
        self._restore_best = restore_best
        self._best_loss = np.infty
        self._best_model_state = None

        self._network = network

    @property
    def patience(self) -> int:
        return self._patience

    def reinitialize(self, network: AbstractArchitecture):
        """
        Reinitializes the EarlyStopping procedure with the same parameters but a new network
        Returns:

        """
        self._network = network
        self._current_patience = self._patience
        self._current_warmup = self._max_warmup
        self._best_loss = np.infty
        self._best_model_state = None

    def __call__(self, loss: Union[float, np.float64]) -> bool:
        """
        Measures if the network improved based on the previous losses. Evaluates to True if it did NOT improve
        for self._patience consecutive iterations.
        If the network improved and self._restore_best, a copy of the network state is created.
        This state is then pasted back into the network if this function evaluates to True
        Args:
            loss: The current evaluation of the loss function. A scalar

        Returns: True if EarlyStopping, False otherwise

        """
        if self._current_warmup >= 0:  # warmup iterations left
            self._current_warmup -= 1
            return False
        if loss < self._best_loss:  # improvement
            if self._restore_best:
                self._best_model_state = deepcopy(self._network.state_dict())
            self._best_loss = loss
            self._current_patience = self._patience
        else:
            self._current_patience -= 1
            if self._current_patience <= 0:
                if self._restore_best:
                    self._network.load_state_dict(self._best_model_state,
                                                  strict=True)  # reset model parameters to best ones
                return True
        return False


def get_early_stopping(early_stopping_config: ConfigDict) -> Union[EarlyStopping, None]:
    early_stopping_patience = early_stopping_config.get("patience", -1)  # disable by default
    assert isinstance(early_stopping_patience, int), f"Need to provide integer patience. " \
                                                     f"Received '{early_stopping_patience} of " \
                                                     f"type '{type(early_stopping_patience)}'"
    if early_stopping_patience >= 0:
        return EarlyStopping(early_stopping_config=early_stopping_config)
    else:
        return None
