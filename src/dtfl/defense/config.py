"""Unified defense configuration.

The framework's privacy knobs live in different layers (m_min in the release
gate, bucket granularity in the transcript, count revelation in emit_record).
``DefenseConfig`` bundles them into one object so an experiment specifies a
single defense point, and the engine/round wiring reads it. The genuinely new
mechanisms (count noising, send-delay jitter, padding) are implemented in their
own modules and parameterized here.

Each knob is documented with its hypothesized effect on the attack (adversary
spec Section 5) and its utility cost, because the headline result is the
trade-off curve over these.

Two axes are PRIMARY (cheap + effective) and are swept for the headline Pareto:
  - ``m_min``                 : anonymity floor (suppression)
  - ``bucket_granularity``    : timing resolution

The rest are single-knob ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["BucketMode", "CountMode", "DefenseConfig"]


class BucketMode(str, Enum):
    """How release times are coarsened into visible buckets."""

    SINGLE = "single"  # one bucket per round: no timing signal (max privacy)
    PER_TIER = "per_tier"  # one bucket per tier window (medium)
    UNIFORM = "uniform"  # uniform-width buckets: granularity set by num_buckets


class CountMode(str, Enum):
    """How the active count is revealed."""

    EXACT = "exact"  # preamble default: reveal n_{k,r} exactly
    HIDDEN = "hidden"  # reveal nothing (count = None); biases the merge
    ROUNDED = "rounded"  # round to a multiple (coarse count)
    NOISED = "noised"  # add bounded/DP noise to the count


@dataclass(frozen=True)
class DefenseConfig:
    """A single point in defense space.

    Defaults = NO defense (the undefended baseline): minimal m_min, finest
    timing, exact counts, no jitter/padding.
    """

    # --- PRIMARY axis 1: anonymity floor (suppression) ---
    m_min: int = 1  # 1 == effectively no suppression
    # number of tiers to use; more tiers => smaller per-tier mass => more
    # suppression bite at a given m_min (interacts with m_min as a privacy knob)
    num_tiers: int = 3

    # --- PRIMARY axis 2: timing resolution ---
    bucket_mode: BucketMode = BucketMode.UNIFORM
    num_buckets: int = 20  # only used for UNIFORM; high == fine == max signal

    # --- count revelation ---
    count_mode: CountMode = CountMode.EXACT
    count_round_to: int = 1  # ROUNDED: multiple to round to
    count_noise_scale: float = 0.0  # NOISED: stddev of additive integer noise

    # --- send-delay jitter (decorrelates fine timing from capability) ---
    send_delay_scale: float = 0.0  # 0 == off; fraction of tier window jittered

    # --- padding (inject dummy contributions to inflate small tiers) ---
    padding_target: int = 0  # 0 == off; pad active count up toward this floor

    @property
    def is_undefended(self) -> bool:
        return (
            self.m_min <= 1
            and self.count_mode is CountMode.EXACT
            and self.send_delay_scale == 0.0
            and self.padding_target == 0
            and not (self.bucket_mode is BucketMode.SINGLE)
        )

    def describe(self) -> str:
        return (
            f"m_min={self.m_min} K={self.num_tiers} "
            f"bucket={self.bucket_mode.value}"
            + (f"/{self.num_buckets}" if self.bucket_mode is BucketMode.UNIFORM else "")
            + f" count={self.count_mode.value}"
            + (f" jitter={self.send_delay_scale}" if self.send_delay_scale else "")
            + (f" pad={self.padding_target}" if self.padding_target else "")
        )
