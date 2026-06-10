"""Timing-resolution adapter.

Builds the right ``Bucketer`` (from dtfl.transcript.bucket) for a DefenseConfig.
The bucket mode is the timing-resolution privacy axis:
  SINGLE   -> no timing signal (max privacy on the timing channel)
  PER_TIER -> one bucket per tier window (medium)
  UNIFORM  -> num_buckets equal-width buckets (granularity = privacy/utility dial)
"""

from __future__ import annotations

from dtfl.defense.config import BucketMode, DefenseConfig
from dtfl.transcript.bucket import (
    Bucketer,
    per_tier_bucket,
    single_bucket,
    uniform_width_bucketer,
)

__all__ = ["bucketer_for"]


def bucketer_for(config: DefenseConfig, round_budget: float) -> Bucketer:
    """Construct the bucketer the engine should use for this defense point.

    Parameters
    ----------
    round_budget:
        The round's final deadline (needed to size UNIFORM buckets). The engine
        knows this after calibrating deadlines.
    """
    mode = config.bucket_mode
    if mode is BucketMode.SINGLE:
        return single_bucket
    if mode is BucketMode.PER_TIER:
        return per_tier_bucket
    if mode is BucketMode.UNIFORM:
        return uniform_width_bucketer(num_buckets=config.num_buckets, round_budget=round_budget)
    raise ValueError(f"unknown bucket mode {mode}")
