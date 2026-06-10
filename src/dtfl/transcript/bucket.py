"""Release-time bucketing.

The server is allowed to see WHEN a tier aggregate is released, but only at a
coarse granularity -- never a raw timestamp (adversary spec Section 2). The
bucket granularity is a DEFENSE KNOB: coarser buckets enlarge the anonymity set
(more participations share a signature) at the cost of timing resolution for the
controller.

Bucketers map a (latent) release time + the round's deadline vector to a small
integer bucket id. This module provides the standard choices; the defense layer
(dtfl.defense.timing_bucket) selects and parameterizes one per experiment.

The bucketer is the ONLY place a continuous latent time is reduced to a visible
quantity. Everything downstream sees only the integer bucket.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dtfl.types import RoundDeadlines

__all__ = [
    "Bucketer",
    "single_bucket",
    "per_tier_bucket",
    "uniform_width_bucketer",
    "quantile_bucketer",
]

# A bucketer maps (release_time, deadlines) -> bucket id (int).
Bucketer = Callable[[float, RoundDeadlines], int]


def single_bucket(_release_time: float, _deadlines: RoundDeadlines) -> int:
    """Maximally coarse: every release in the round maps to bucket 0.

    No timing signal at all. The safest default for the privacy claim; the
    controller then has no within-round timing resolution (it still sees counts).
    """
    return 0


def per_tier_bucket(release_time: float, deadlines: RoundDeadlines) -> int:
    """One bucket per tier window: bucket = index of the tier whose deadline the
    release falls under. Reveals roughly which tier released when, no finer.

    This is the natural medium-granularity choice: the server already learns the
    tier structure, so bucketing to tier windows leaks little beyond what the
    tier identity (visible on the record anyway) already implies.
    """
    cutoffs = np.asarray(deadlines.cutoffs, dtype=np.float64)
    idx = int(np.searchsorted(cutoffs, release_time, side="left"))
    return min(idx, cutoffs.size - 1)


def uniform_width_bucketer(num_buckets: int, round_budget: float) -> Bucketer:
    """Divide [0, round_budget] into ``num_buckets`` equal-width buckets.

    Finer timing resolution -> stronger linkability signal for the attacker.
    Sweeping ``num_buckets`` from 1 (== single_bucket) up to many traces the
    privacy/utility curve along the timing axis.
    """
    if num_buckets < 1:
        raise ValueError("num_buckets must be >= 1")
    if round_budget <= 0:
        raise ValueError("round_budget must be > 0")
    width = round_budget / num_buckets

    def _bucketer(release_time: float, _deadlines: RoundDeadlines) -> int:
        b = int(release_time // width)
        return max(0, min(b, num_buckets - 1))

    return _bucketer


def quantile_bucketer(edges: tuple[float, ...]) -> Bucketer:
    """Bucket by pre-specified time edges (e.g. learned timing quantiles).

    ``edges`` are increasing cut points; a release time falls in the bucket of
    the first edge it does not exceed. Lets the defense match bucket boundaries
    to the actual completion-time distribution rather than uniform spacing.
    """
    edge_arr = np.asarray(edges, dtype=np.float64)
    if edge_arr.size and np.any(np.diff(edge_arr) <= 0):
        raise ValueError("edges must be strictly increasing")

    def _bucketer(release_time: float, _deadlines: RoundDeadlines) -> int:
        return int(np.searchsorted(edge_arr, release_time, side="left"))

    return _bucketer
