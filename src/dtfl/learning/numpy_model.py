"""Pure-NumPy reference FL model (softmax classifier / 1-hidden-layer MLP).

Implements FLModel with no torch dependency, so the entire training pipeline --
including time-to-accuracy -- runs anywhere for fast iteration and for CI where
torch is unavailable. The torch model (torch_model.py) implements the identical
interface for the real CIFAR/FEMNIST results; the FL loop is backend-identical.

Multinomial logistic regression by default; set hidden_dim > 0 for a 1-hidden
ReLU layer. Local update is plain mini-batch SGD on cross-entropy. The returned
delta is (w_after_local_sgd - w_global) flattened, which is exactly the per-
client update the secure-aggregation sum and FedAvg-equivalent merge consume.
"""

from __future__ import annotations

import numpy as np

from dtfl.learning.model import Dataset, FLModel

__all__ = ["NumpySoftmaxModel"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NumpySoftmaxModel(FLModel):
    """Softmax classifier (optionally with one hidden ReLU layer)."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 0, seed: int = 0):
        self._d_in = input_dim
        self._c = num_classes
        self._h = hidden_dim
        rng = np.random.default_rng(seed)
        scale = 0.01
        if hidden_dim > 0:
            self._W1 = rng.normal(0, scale, (input_dim, hidden_dim))
            self._b1 = np.zeros(hidden_dim)
            self._W2 = rng.normal(0, scale, (hidden_dim, num_classes))
            self._b2 = np.zeros(num_classes)
        else:
            self._W = rng.normal(0, scale, (input_dim, num_classes))
            self._b = np.zeros(num_classes)

    # --- flat param packing ---

    def get_params(self) -> np.ndarray:
        if self._h > 0:
            return np.concatenate([
                self._W1.ravel(), self._b1, self._W2.ravel(), self._b2
            ])
        return np.concatenate([self._W.ravel(), self._b])

    def set_params(self, params: np.ndarray) -> None:
        params = np.asarray(params, dtype=np.float64)
        if self._h > 0:
            i = 0
            n1 = self._d_in * self._h
            self._W1 = params[i:i + n1].reshape(self._d_in, self._h); i += n1
            self._b1 = params[i:i + self._h]; i += self._h
            n2 = self._h * self._c
            self._W2 = params[i:i + n2].reshape(self._h, self._c); i += n2
            self._b2 = params[i:i + self._c]
        else:
            n = self._d_in * self._c
            self._W = params[:n].reshape(self._d_in, self._c)
            self._b = params[n:n + self._c]

    @property
    def dim(self) -> int:
        if self._h > 0:
            return self._d_in * self._h + self._h + self._h * self._c + self._c
        return self._d_in * self._c + self._c

    # --- forward / loss ---

    def _forward(self, X: np.ndarray, params: dict | None = None):
        if self._h > 0:
            Z1 = X @ self._W1 + self._b1
            A1 = np.maximum(0, Z1)
            logits = A1 @ self._W2 + self._b2
            return logits, A1
        return X @ self._W + self._b, None

    def evaluate(self, data: Dataset) -> tuple[float, float]:
        logits, _ = self._forward(data.X)
        probs = _softmax(logits)
        n = len(data)
        ll = -np.log(probs[np.arange(n), data.y] + 1e-12).mean()
        acc = float((probs.argmax(axis=1) == data.y).mean())
        return float(ll), acc

    # --- local SGD; returns delta, does not mutate stored global params ---

    def local_update(
        self,
        data: Dataset,
        local_epochs: int,
        lr: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        w_global = self.get_params().copy()  # snapshot to restore + diff
        n = len(data)
        for _ in range(local_epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                bidx = order[start:start + batch_size]
                Xb, yb = data.X[bidx], data.y[bidx]
                self._sgd_step(Xb, yb, lr)
        delta = self.get_params() - w_global
        self.set_params(w_global)  # restore: caller manages the global model
        return delta

    def _sgd_step(self, Xb: np.ndarray, yb: np.ndarray, lr: float) -> None:
        b = Xb.shape[0]
        if self._h > 0:
            Z1 = Xb @ self._W1 + self._b1
            A1 = np.maximum(0, Z1)
            logits = A1 @ self._W2 + self._b2
            probs = _softmax(logits)
            probs[np.arange(b), yb] -= 1.0
            dlogits = probs / b
            gW2 = A1.T @ dlogits
            gb2 = dlogits.sum(axis=0)
            dA1 = dlogits @ self._W2.T
            dZ1 = dA1 * (Z1 > 0)
            gW1 = Xb.T @ dZ1
            gb1 = dZ1.sum(axis=0)
            self._W2 -= lr * gW2; self._b2 -= lr * gb2
            self._W1 -= lr * gW1; self._b1 -= lr * gb1
        else:
            logits = Xb @ self._W + self._b
            probs = _softmax(logits)
            probs[np.arange(b), yb] -= 1.0
            dlogits = probs / b
            gW = Xb.T @ dlogits
            gb = dlogits.sum(axis=0)
            self._W -= lr * gW; self._b -= lr * gb
