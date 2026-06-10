"""dtfl.learning: real FL training. NEVER importable from dtfl.attack.

TorchModel is imported lazily (only when requested) so the package imports
without torch installed; the NumPy model and the FL loop have no torch dependency.
"""
from dtfl.learning.datasets import iid_shard, load_real_dataset, make_synthetic_classification
from dtfl.learning.federated import FedTrainConfig, FedTrainResult, federated_train
from dtfl.learning.model import Dataset, FLModel
from dtfl.learning.numpy_model import NumpySoftmaxModel


def resolve_device(device: str = "auto") -> str:
    """Resolve "auto" to cuda when available, else cpu. Safe if torch absent."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_torch_model(*args, **kwargs):
    """Lazy accessor for the torch model (imports torch only when called)."""
    from dtfl.learning.torch_model import TorchModel
    return TorchModel(*args, **kwargs)


__all__ = [
    "Dataset", "FLModel", "NumpySoftmaxModel", "get_torch_model",
    "iid_shard", "make_synthetic_classification", "load_real_dataset",
    "resolve_device",
    "FedTrainConfig", "FedTrainResult", "federated_train",
]
