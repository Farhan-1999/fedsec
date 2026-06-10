"""Fixed-deadline controllers (the baselines).

Two non-adaptive policies from the experimental design:
  - FixedEqualWidth: uniform spacing over a round budget. The WEAK baseline.
  - FixedQuantile: deadlines frozen at quantiles estimated from a warm-up pilot.
    The STRONG simple baseline -- it matches the completion-time distribution
    once, then never adapts.

Both ignore the transcript view (they are transcript-blind by construction);
FixedQuantile's quantiles come from an experimenter pilot using latent draws,
which is a one-time calibration, not a per-round read of hidden state. The
resulting policy at run time still reads nothing.
"""

from __future__ import annotations

from dtfl.controller.base import Controller
from dtfl.transcript.store import TranscriptView

__all__ = ["FixedEqualWidth", "FixedQuantile"]


class FixedEqualWidth(Controller):
    """Uniform spacing over [0, round_budget]. The weak baseline."""

    def __init__(self, num_tiers: int, round_budget: float):
        super().__init__(num_tiers)
        self._budget = round_budget
        # equal-width cutoffs: budget * (1/K, 2/K, ..., 1)
        self._cutoffs = tuple(
            round_budget * (k + 1) / num_tiers for k in range(num_tiers)
        )

    def next_deadlines(self, round_index: int, view: TranscriptView) -> tuple[float, ...]:
        return self._cutoffs


class FixedQuantile(Controller):
    """Deadlines frozen at pilot-estimated completion-time quantiles.

    The cutoffs are supplied by the experimenter (computed once from a latent
    pilot via Engine.calibrate_fixed_deadlines). The controller just replays
    them every round -- a strong, stationary baseline.
    """

    def __init__(self, num_tiers: int, cutoffs: tuple[float, ...]):
        super().__init__(num_tiers)
        if len(cutoffs) != num_tiers:
            raise ValueError(f"cutoffs length {len(cutoffs)} != num_tiers {num_tiers}")
        self._cutoffs = tuple(float(c) for c in cutoffs)

    def next_deadlines(self, round_index: int, view: TranscriptView) -> tuple[float, ...]:
        return self._cutoffs
