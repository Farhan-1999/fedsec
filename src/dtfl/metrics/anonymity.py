"""Anonymity-set size from observable signatures (adversary spec Goal B).

A device's visible SIGNATURE over a horizon h is the sequence of (tier,
release-bucket) it produced in released tier-rounds up to round h. Two devices
are INDISTINGUISHABLE over h if their signatures are identical. The equivalence
class E_i(h) is the set of devices sharing device i's signature; its size
|E_i(h)| is the realized anonymity set.

Privacy reading:
  - |E_i(h)| large  -> device i hides in a crowd of look-alikes (good).
  - |E_i(h)| == 1   -> device i is uniquely identifiable by its signature (bad).
  - fraction with |E_i(h)| < m_min -> the privacy-violation rate.

Because each additional round adds to the signature, |E_i(h)| is NON-INCREASING
in h: more observation can only split crowds, never merge them. That monotone
shrinkage is the "leakage accumulates over rounds" mechanism, quantified in
linkability.py via L_i(h) = 1/|E_i(h)|.

These functions take per-device observable signatures (built by the harness from
the released-tier-filtered observations -- the same channel the attacker uses).
They never read capability class; anonymity is a property of the visible
signature alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

__all__ = ["AnonymityStats", "signature_up_to", "equivalence_class_sizes", "anonymity_stats"]

# A per-round signature atom is (tier_index, release_bucket). A device's signature
# over a horizon is the tuple of its atoms in round order. Rounds where the device
# did not appear in a released tier contribute a "gap" atom so absence is itself
# part of the signature (an attacker notices a device going quiet).
GAP = (-1, -1)


def signature_up_to(
    per_round_atom: dict[int, tuple[int, int]],
    horizon: int,
) -> tuple[tuple[int, int], ...]:
    """Build a device's signature over rounds 0..horizon-1.

    Parameters
    ----------
    per_round_atom:
        Map round_index -> (tier, bucket) for rounds the device appeared in a
        released tier. Missing rounds become GAP atoms.
    horizon:
        Number of rounds to include (signature length is exactly ``horizon``).
    """
    return tuple(per_round_atom.get(r, GAP) for r in range(horizon))


def equivalence_class_sizes(
    device_signatures: dict[int, tuple],
) -> dict[int, int]:
    """Map each device id to the size of its signature equivalence class.

    Devices with identical signatures share a class; the size is how many devices
    carry that signature (including the device itself).
    """
    sig_counts = Counter(device_signatures.values())
    return {did: sig_counts[sig] for did, sig in device_signatures.items()}


@dataclass
class AnonymityStats:
    """Distribution summary of |E_i(h)| at a given horizon."""

    horizon: int
    num_devices: int
    minimum: int
    p5: float
    median: float
    mean: float
    fraction_unique: float  # fraction with |E_i(h)| == 1 (fully identifiable)
    fraction_below_m_min: float  # privacy-violation rate

    def summary(self) -> str:
        return (
            f"h={self.horizon:>3} n={self.num_devices:>4} "
            f"min={self.minimum} p5={self.p5:.1f} med={self.median:.1f} "
            f"unique={100*self.fraction_unique:.1f}% "
            f"<m_min={100*self.fraction_below_m_min:.1f}%"
        )


def anonymity_stats(
    class_sizes: dict[int, int],
    m_min: int,
) -> AnonymityStats:
    """Summarize an equivalence-class-size distribution."""
    sizes = np.array(list(class_sizes.values()), dtype=np.float64)
    if sizes.size == 0:
        return AnonymityStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return AnonymityStats(
        horizon=-1,  # set by caller
        num_devices=sizes.size,
        minimum=int(sizes.min()),
        p5=float(np.percentile(sizes, 5)),
        median=float(np.median(sizes)),
        mean=float(sizes.mean()),
        fraction_unique=float((sizes == 1).mean()),
        fraction_below_m_min=float((sizes < m_min).mean()),
    )
