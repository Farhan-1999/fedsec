"""IID data partitioning for the federated population.

The mainline regime (preamble Section 2, PETS dataset-relevance requirement)
pools each benchmark centrally and RE-SHARDS IID across synthetic clients; the
natural federated partitions are confined to a sensitivity appendix. This module
provides:
  - the IID sharding logic (the part that must be correct and is tested here),
  - a synthetic separable-Gaussian classification dataset for verification / CI
    where the real benchmarks are unavailable, and
  - documented hooks for the real loaders (CIFAR-10/100, FEMNIST), which run on a
    machine with torchvision/LEAF and are wired in via the same Dataset return
    type.

IID sharding: pool all (X, y), shuffle, split into ``num_clients`` roughly equal
shards. Because the split is uniform-random over the pooled set, each client's
shard is an IID sample of the global distribution -- which is exactly what makes
the single-global-model merge and FedAvg-equivalence valid (preamble Section 2).
"""

from __future__ import annotations

import numpy as np

from dtfl.learning.model import Dataset

__all__ = ["iid_shard", "make_synthetic_classification", "load_real_dataset"]


def iid_shard(
    data: Dataset,
    num_clients: int,
    rng: np.random.Generator,
) -> list[Dataset]:
    """Pool-and-reshard IID: uniform-random equal-ish shards over the pooled set.

    Each client shard is an IID draw from the global distribution. Shards differ
    in size by at most one sample.
    """
    n = len(data)
    perm = rng.permutation(n)
    shards = np.array_split(perm, num_clients)
    return [data.subset(idx) for idx in shards]


def make_synthetic_classification(
    num_samples: int,
    num_features: int,
    num_classes: int,
    rng: np.random.Generator,
    separation: float = 2.0,
) -> tuple[Dataset, Dataset]:
    """Separable-Gaussian classification data (train, val). Verification stand-in.

    Each class is a Gaussian blob with mean at ``separation`` * a random unit
    direction. Linearly (mostly) separable so a softmax classifier reaches high
    accuracy quickly -- enough to exercise the full time-to-accuracy loop and the
    FedAvg-equivalence check without a real dataset.
    """
    centers = rng.normal(0, 1, (num_classes, num_features))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= separation

    def draw(n):
        y = rng.integers(0, num_classes, size=n)
        X = centers[y] + rng.normal(0, 1, (n, num_features))
        return Dataset(X, y)

    train = draw(num_samples)
    val = draw(max(200, num_samples // 5))
    return train, val


def load_real_dataset(name: str, data_root: str = "./data") -> tuple[Dataset, Dataset]:
    """Load and pool a real benchmark, returning (train, val) as flat-feature Datasets.

    Runs on a machine with torchvision / LEAF available. Pools the standard
    train/test split centrally (the IID resharding across clients is done
    afterward by ``iid_shard``). Implemented for the NumPy model as flattened,
    normalized features; the torch model can instead consume image tensors via a
    parallel loader.

    Supported names (when deps present): "cifar10", "cifar100", "femnist".
    Raises ImportError with guidance if torchvision is missing.
    """
    name = name.lower()
    try:
        import torchvision  # noqa: F401
        import torchvision.transforms as T
        from torchvision import datasets as tvd
    except ImportError as e:  # pragma: no cover - exercised only off-sandbox
        raise ImportError(
            "Real datasets need torchvision: pip install '.[learning]'. "
            "For CI / offline use, use make_synthetic_classification instead."
        ) from e

    def to_flat(ds, max_n=None):  # pragma: no cover - needs torchvision
        Xs, ys = [], []
        for i, (img, label) in enumerate(ds):
            if max_n and i >= max_n:
                break
            Xs.append(np.asarray(img).reshape(-1))
            ys.append(int(label))
        X = np.stack(Xs).astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)  # standardize features
        return Dataset(X, np.array(ys))

    tfm = T.ToTensor()
    if name == "cifar10":  # pragma: no cover
        tr = tvd.CIFAR10(data_root, train=True, download=True, transform=tfm)
        te = tvd.CIFAR10(data_root, train=False, download=True, transform=tfm)
    elif name == "cifar100":  # pragma: no cover
        tr = tvd.CIFAR100(data_root, train=True, download=True, transform=tfm)
        te = tvd.CIFAR100(data_root, train=False, download=True, transform=tfm)
    else:  # pragma: no cover
        raise ValueError(f"unsupported dataset {name} (use cifar10/cifar100, or FEMNIST via LEAF)")
    return to_flat(tr), to_flat(te)
