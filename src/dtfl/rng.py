"""Seeded randomness, threaded from a single root.

Every stochastic draw in the simulator MUST come from a Generator produced here.
This is what makes a run reproducible: one root seed determines the entire
latent population AND the entire transcript. Baselines share a fixed latent
trace by sharing the same root seed, so differences between methods are
attributable to the mechanism, not to a different random population.

Design:
- One root seed (int) per experiment, set in config.
- Named sub-streams (``spawn``) so independent components don't share state and
  reordering one component's draws can't perturb another's. Adding a new
  component never shifts the random draws of existing ones, which keeps old
  results reproducible as the codebase grows.

Never call ``numpy.random`` module-level functions anywhere in ``dtfl``; they use
a hidden global state that breaks reproducibility. Always pass a Generator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = ["RngHub", "make_root", "stable_substream_seed"]


def stable_substream_seed(root_seed: int, name: str) -> int:
    """Derive a deterministic sub-stream seed from a root seed and a name.

    Uses a hash of the name so that the seed for a stream depends only on its
    name and the root, never on the order in which streams are requested. This
    is the property that keeps existing components' draws stable when new
    components are added.
    """
    h = hashlib.sha256(f"{root_seed}:{name}".encode()).digest()
    # Take 8 bytes -> 64-bit int, then mask to NumPy's accepted seed range.
    return int.from_bytes(h[:8], "big") % (2**32 - 1)


def make_root(seed: int) -> np.random.Generator:
    """Create the root Generator for an experiment from an integer seed."""
    return np.random.default_rng(seed)


@dataclass
class RngHub:
    """Hands out named, independent Generators derived from one root seed.

    Usage:
        hub = RngHub(seed=12345)
        dev_rng = hub.stream("latent.population")
        lat_rng = hub.stream("latent.latency")

    Requesting the same name twice returns generators seeded identically;
    request a name once and hold the Generator if you need a single evolving
    stream. Distinct names are statistically independent.
    """

    seed: int

    def stream(self, name: str) -> np.random.Generator:
        """Return a Generator for the named sub-stream."""
        return np.random.default_rng(stable_substream_seed(self.seed, name))

    def child(self, name: str) -> RngHub:
        """Return a sub-hub whose own streams are namespaced under ``name``.

        Useful for per-round or per-device namespacing, e.g.
        ``hub.child(f"round.{r}").stream("dropout")``.
        """
        return RngHub(seed=stable_substream_seed(self.seed, name))
