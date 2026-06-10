"""dtfl.latent: ground-truth generative model. NEVER importable from dtfl.attack."""
from dtfl.latent.config import DriftConfig, LatentConfig
from dtfl.latent.device import make_device
from dtfl.latent.drift import DriftState
from dtfl.latent.latency import (
    LatencyDecomposition,
    decompose,
    draw_completion_times,
)
from dtfl.latent.population import (
    build_population,
    class_tier_mutual_information,
    draw_classes,
)

__all__ = [
    "LatentConfig",
    "DriftConfig",
    "make_device",
    "DriftState",
    "draw_completion_times",
    "decompose",
    "LatencyDecomposition",
    "build_population",
    "draw_classes",
    "class_tier_mutual_information",
]
