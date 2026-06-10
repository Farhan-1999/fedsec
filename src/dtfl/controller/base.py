"""Deadline-controller interface.

A controller decides the tier deadline vector each round. It is the SYSTEM BEING
MEASURED: the privacy/utility results are reported as a function of which
controller sets the deadlines. The engine consumes a ``DeadlinePolicy`` callable
(round_index, transcript_view) -> cutoffs; a controller is a stateful object that
exposes such a callable via ``policy()``.

CRITICAL separation rule: a REAL (deployable) controller may read ONLY the
transcript view -- per-tier counts, release buckets, success flags -- exactly the
aggregate-only observables the server is allowed (preamble Section 1: the server
acts on population-level tier statistics, never per-device state). The oracle
controller is the deliberate exception (it reads latent ground truth as an
upper-bound reference) and lives in controller/oracle.py, which the attack
package is forbidden to import.

All controllers must return strictly increasing cutoffs of the configured length
(the engine validates this); a projection helper enforces monotonicity + spacing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from dtfl.transcript.store import TranscriptView

__all__ = ["Controller", "project_monotone", "DeadlinePolicy"]

DeadlinePolicy = Callable[[int, TranscriptView], tuple[float, ...]]


def project_monotone(
    cutoffs: np.ndarray,
    min_spacing: float,
    lo: float,
    hi: float,
) -> tuple[float, ...]:
    """Force a deadline vector to be increasing with >= min_spacing, within [lo, hi].

    Left-to-right clamp: each cutoff is pushed to at least previous + min_spacing
    and clamped to [lo, hi]. Guarantees strict monotonicity the engine requires.
    """
    out = np.array(cutoffs, dtype=np.float64)
    out[0] = np.clip(out[0], lo, hi)
    for k in range(1, out.size):
        out[k] = np.clip(out[k], out[k - 1] + min_spacing, hi)
    return tuple(float(x) for x in out)


class Controller(ABC):
    """Stateful deadline controller. Exposes a DeadlinePolicy via ``policy()``."""

    def __init__(self, num_tiers: int):
        self.num_tiers = num_tiers

    @abstractmethod
    def next_deadlines(self, round_index: int, view: TranscriptView) -> tuple[float, ...]:
        """Return the deadline vector for ``round_index`` given the transcript so far."""

    def policy(self) -> DeadlinePolicy:
        """Return the callable the engine consumes."""
        return self.next_deadlines
