"""Backend-agnostic model interface for federated training.

The FL loop only needs four operations from a model:
  - get_params() / set_params(): read/write the global weight vector,
  - local_update(data): run local SGD from the current weights, return the
    DELTA (w_local - w_global) as a flat vector -- exactly the per-client update
    the secure-aggregation sum and the FedAvg-equivalent merge operate on,
  - evaluate(data): return (loss, accuracy) on a validation set.

By programming the FL loop against this interface, the simulator is independent
of the backend. We provide a pure-NumPy reference model (logistic regression /
1-hidden-layer MLP) that runs anywhere with no torch, and a torch model for the
real CIFAR/FEMNIST results. Both return updates in the SAME flat-vector form, so
the merge math, the FedAvg-equivalence check, and the time-to-accuracy loop are
backend-identical.

The delta vector is what flows into TierContribution.secure_sum (replacing the
synthetic vector used in Steps 0-2). Nothing else in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["FLModel", "Dataset"]


class Dataset:
    """Minimal (X, y) container; X is (n, d) features, y is (n,) integer labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def subset(self, idx: np.ndarray) -> Dataset:
        return Dataset(self.X[idx], self.y[idx])


class FLModel(ABC):
    """Backend-agnostic federated model."""

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return the current parameters as a flat float vector."""

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Set parameters from a flat float vector."""

    @abstractmethod
    def local_update(
        self,
        data: Dataset,
        local_epochs: int,
        lr: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Run local SGD from current params; return the delta (w_local - w_global).

        Must NOT mutate the model's own params (the caller restores the global
        params between clients). Returns a flat vector the same length as
        get_params().
        """

    @abstractmethod
    def evaluate(self, data: Dataset) -> tuple[float, float]:
        """Return (cross-entropy loss, accuracy) on ``data``."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Number of parameters (length of the flat vector)."""
