"""Torch FL model for the real CIFAR/FEMNIST results.

Implements the identical FLModel interface as the NumPy reference, so the FL loop
is backend-identical: only the model construction line changes. Runs on a machine
with torch installed (pip install '.[learning]'); not exercised in the offline
sandbox, hence the import is local to construction.

Provides a small CNN (for image tasks) and an MLP, selectable by ``arch``. The
flat-vector get/set_params packing mirrors the NumPy model so deltas are
interchangeable in the merge.
"""

from __future__ import annotations

import numpy as np

from dtfl.learning.model import Dataset, FLModel

__all__ = ["TorchModel"]


class TorchModel(FLModel):  # pragma: no cover - requires torch, runs off-sandbox
    """Torch-backed model (MLP or small CNN) behind the FLModel interface."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        arch: str = "mlp",
        hidden_dim: int = 128,
        image_shape: tuple[int, int, int] | None = None,
        seed: int = 0,
        device: str = "auto",
    ):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self._torch = torch
        self._nn = nn
        # "auto" -> use CUDA when present, else CPU. This is the only place device
        # is resolved; the whole model (net + tensors) then lives on self._device.
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._num_classes = num_classes
        self._image_shape = image_shape

        if arch == "mlp":
            self._net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            ).to(self._device)
        elif arch == "cnn":
            if image_shape is None:
                raise ValueError("cnn arch requires image_shape=(C,H,W)")
            c, h, w = image_shape
            self._net = nn.Sequential(
                nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * (h // 4) * (w // 4), num_classes),
            ).to(self._device)
        else:
            raise ValueError(f"unknown arch {arch}")

        self._loss = nn.CrossEntropyLoss()

    # --- flat param packing (mirrors NumPy model semantics) ---

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in self._net.parameters()
        ])

    def set_params(self, params: np.ndarray) -> None:
        torch = self._torch
        i = 0
        for p in self._net.parameters():
            n = p.numel()
            chunk = params[i:i + n].reshape(p.shape)
            p.data = torch.tensor(chunk, dtype=p.dtype, device=self._device)
            i += n

    @property
    def dim(self) -> int:
        return sum(p.numel() for p in self._net.parameters())

    def _to_tensor(self, data: Dataset):
        torch = self._torch
        X = data.X
        if self._image_shape is not None:
            X = X.reshape((-1, *self._image_shape))
        Xt = torch.tensor(X, dtype=torch.float32, device=self._device)
        yt = torch.tensor(data.y, dtype=torch.long, device=self._device)
        return Xt, yt

    def evaluate(self, data: Dataset) -> tuple[float, float]:
        torch = self._torch
        self._net.eval()
        Xt, yt = self._to_tensor(data)
        with torch.no_grad():
            logits = self._net(Xt)
            loss = float(self._loss(logits, yt).item())
            acc = float((logits.argmax(1) == yt).float().mean().item())
        return loss, acc

    def local_update(
        self,
        data: Dataset,
        local_epochs: int,
        lr: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        torch = self._torch
        w_global = self.get_params().copy()
        self._net.train()
        opt = torch.optim.SGD(self._net.parameters(), lr=lr)
        Xt, yt = self._to_tensor(data)
        n = len(data)
        for _ in range(local_epochs):
            order = torch.tensor(rng.permutation(n), device=self._device)
            for start in range(0, n, batch_size):
                bidx = order[start:start + batch_size]
                opt.zero_grad()
                logits = self._net(Xt[bidx])
                loss = self._loss(logits, yt[bidx])
                loss.backward()
                opt.step()
        delta = self.get_params() - w_global
        self.set_params(w_global)
        return delta
