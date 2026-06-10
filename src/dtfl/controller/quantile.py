"""Quantile-tracking deadline controller (Robbins-Monro stochastic approximation).

The recommended ADAPTIVE controller (control document). Goal: place tier deadline
d_k so that a target fraction q_k of participants completes by it -- i.e. track
the q_k-quantile of the (unobserved) completion-time distribution using ONLY the
aggregate-only transcript.

Each round it reads per-tier counts from the transcript, forms the empirical CDF
at the current cutoffs (cumulative participant fraction up to each tier), and
nudges each deadline by a Robbins-Monro step toward its target quantile:

    d_k <- d_k + eta_r * (q_k - F_hat(d_k)) / max(eps, f_hat(d_k))

where F_hat is the observed cumulative fraction and f_hat is a finite-difference
density proxy. Step sizes eta_r decay (eta0/(r+1)) for stochastic-approximation
stability under stationarity; a projection enforces monotonic, spaced cutoffs.

Reads ONLY the transcript view (counts) -- it never sees a completion time or a
device. This is what makes it a deployable, privacy-respecting controller, unlike
per-client schedulers (FedCS/Oort).
"""

from __future__ import annotations

import numpy as np

from dtfl.controller.base import Controller, project_monotone
from dtfl.transcript.store import TranscriptView

__all__ = ["QuantileTrackingController"]


class QuantileTrackingController(Controller):
    def __init__(
        self,
        num_tiers: int,
        targets: tuple[float, ...],
        init_cutoffs: tuple[float, ...],
        *,
        eta0: float = 0.5,
        min_spacing: float = 0.05,
        lo: float = 0.0,
        hi: float | None = None,
        max_step: float | None = None,
    ):
        """
        targets:
            Target cumulative fractions q_1 < ... < q_K. The last is typically
            1.0 (the final cutoff covers everyone). Length == num_tiers.
        init_cutoffs:
            Starting deadline vector (e.g. equal-width or a rough pilot).
        eta0:
            Base step size; per-round step is eta0/(round+1) (Robbins-Monro decay).
        """
        super().__init__(num_tiers)
        if len(targets) != num_tiers or len(init_cutoffs) != num_tiers:
            raise ValueError("targets and init_cutoffs must have length num_tiers")
        self._q = np.array(targets, dtype=np.float64)
        self._cutoffs = np.array(init_cutoffs, dtype=np.float64)
        self._eta0 = eta0
        self._min_spacing = min_spacing
        self._lo = lo
        self._hi = hi if hi is not None else float(init_cutoffs[-1]) * 3.0
        self._max_step = max_step if max_step is not None else float(init_cutoffs[-1])

    def _empirical_cdf(self, view: TranscriptView, round_index: int) -> np.ndarray:
        """Cumulative participant fraction up to each tier, from transcript counts.

        F_hat[k] = (sum of counts in tiers 0..k) / (total counts this round).
        Suppressed tiers contribute None -> treated as 0 here (the server cannot
        count a suppressed tier's participants); this biases F downward when
        suppression is active, which the controller experiences as "fewer arrived"
        and responds to by widening -- a sensible reaction.
        """
        counts = view.tier_counts(round_index)
        per_tier = np.array(
            [(counts.get(k) or 0) for k in range(self.num_tiers)], dtype=np.float64
        )
        total = per_tier.sum()
        if total <= 0:
            return np.full(self.num_tiers, np.nan)
        return np.cumsum(per_tier) / total

    def next_deadlines(self, round_index: int, view: TranscriptView) -> tuple[float, ...]:
        # First round (no transcript yet): emit the init cutoffs.
        if round_index == 0 or len(view) == 0:
            return project_monotone(self._cutoffs, self._min_spacing, self._lo, self._hi)

        prev = round_index - 1
        F = self._empirical_cdf(view, prev)
        if np.all(np.isnan(F)):
            return project_monotone(self._cutoffs, self._min_spacing, self._lo, self._hi)

        eta = self._eta0 / (round_index + 1)
        new = self._cutoffs.copy()
        for k in range(self.num_tiers):
            if np.isnan(F[k]):
                continue
            # finite-difference density proxy: dF/dd around tier k
            k_lo = max(0, k - 1)
            k_hi = min(self.num_tiers - 1, k + 1)
            dF = F[k_hi] - F[k_lo]
            dd = self._cutoffs[k_hi] - self._cutoffs[k_lo]
            f_hat = max(1e-3, dF / dd) if dd > 0 else 1e-3
            step = eta * (self._q[k] - F[k]) / f_hat
            step = float(np.clip(step, -self._max_step, self._max_step))
            new[k] = self._cutoffs[k] + step

        self._cutoffs = np.array(
            project_monotone(new, self._min_spacing, self._lo, self._hi)
        )
        return tuple(float(x) for x in self._cutoffs)
