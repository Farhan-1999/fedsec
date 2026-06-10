"""EWMA-smoothed quantile controller for drifting regimes.

Same quantile-tracking idea as QuantileTrackingController, but it smooths the
observed empirical CDF with an exponentially weighted moving average before
stepping, and uses a constant (not decaying) step size capped per round. This
trades the stationary-convergence guarantee of Robbins-Monro for responsiveness
to nonstationary latency regimes (regime shifts, drift) -- the setting the drift
layer models.

Like the plain quantile controller, it reads ONLY the transcript counts.
"""

from __future__ import annotations

import numpy as np

from dtfl.controller.base import Controller, project_monotone
from dtfl.transcript.store import TranscriptView

__all__ = ["EWMAQuantileController"]


class EWMAQuantileController(Controller):
    def __init__(
        self,
        num_tiers: int,
        targets: tuple[float, ...],
        init_cutoffs: tuple[float, ...],
        *,
        beta: float = 0.2,
        eta: float = 0.3,
        min_spacing: float = 0.05,
        lo: float = 0.0,
        hi: float | None = None,
        max_step: float | None = None,
    ):
        """
        beta: EWMA weight on the newest observation (higher = more responsive).
        eta:  constant step size (no Robbins-Monro decay; capped by max_step).
        """
        super().__init__(num_tiers)
        self._q = np.array(targets, dtype=np.float64)
        self._cutoffs = np.array(init_cutoffs, dtype=np.float64)
        self._beta = beta
        self._eta = eta
        self._min_spacing = min_spacing
        self._lo = lo
        self._hi = hi if hi is not None else float(init_cutoffs[-1]) * 3.0
        self._max_step = max_step if max_step is not None else float(init_cutoffs[-1]) * 0.2
        self._F_ewma: np.ndarray | None = None  # smoothed empirical CDF

    def _observed_cdf(self, view: TranscriptView, round_index: int) -> np.ndarray:
        counts = view.tier_counts(round_index)
        per_tier = np.array(
            [(counts.get(k) or 0) for k in range(self.num_tiers)], dtype=np.float64
        )
        total = per_tier.sum()
        if total <= 0:
            return np.full(self.num_tiers, np.nan)
        return np.cumsum(per_tier) / total

    def next_deadlines(self, round_index: int, view: TranscriptView) -> tuple[float, ...]:
        if round_index == 0 or len(view) == 0:
            return project_monotone(self._cutoffs, self._min_spacing, self._lo, self._hi)

        F_obs = self._observed_cdf(view, round_index - 1)
        if np.all(np.isnan(F_obs)):
            return project_monotone(self._cutoffs, self._min_spacing, self._lo, self._hi)

        # EWMA update of the smoothed CDF.
        if self._F_ewma is None:
            self._F_ewma = np.nan_to_num(F_obs, nan=0.0)
        else:
            obs = np.nan_to_num(F_obs, nan=0.0)
            self._F_ewma = (1 - self._beta) * self._F_ewma + self._beta * obs

        F = self._F_ewma
        new = self._cutoffs.copy()
        for k in range(self.num_tiers):
            k_lo = max(0, k - 1)
            k_hi = min(self.num_tiers - 1, k + 1)
            dF = F[k_hi] - F[k_lo]
            dd = self._cutoffs[k_hi] - self._cutoffs[k_lo]
            f_hat = max(1e-3, dF / dd) if dd > 0 else 1e-3
            step = self._eta * (self._q[k] - F[k]) / f_hat
            step = float(np.clip(step, -self._max_step, self._max_step))
            new[k] = self._cutoffs[k] + step

        self._cutoffs = np.array(
            project_monotone(new, self._min_spacing, self._lo, self._hi)
        )
        return tuple(float(x) for x in self._cutoffs)
