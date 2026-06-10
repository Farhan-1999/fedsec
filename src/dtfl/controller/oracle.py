"""Oracle deadline controller (upper-bound reference).

THIS MODULE READS LATENT GROUND TRUTH. It is the deliberate exception to the
observation boundary: it sets deadlines directly from the TRUE completion-time
CDF, so it represents the best any deadline policy could do if the server somehow
knew the latency distribution. It is the upper bound against which the realistic
transcript-only controllers (quantile, EWMA) are measured -- the "price of
privacy" is the gap between oracle and adaptive.

Because it reads latent state, this module MUST NOT be reachable from
dtfl.attack. The separation tests list dtfl.controller.oracle in the forbidden
set; the attack package's import closure is checked against it. Do not import
this module from any attacker code.

It is NOT a deployable controller -- a real server cannot see the true CDF. It
exists only as an experimental reference line.
"""

from __future__ import annotations

import numpy as np

from dtfl.controller.base import Controller, project_monotone
from dtfl.latent.config import LatentConfig
from dtfl.latent.latency import draw_completion_times
from dtfl.transcript.store import TranscriptView

__all__ = ["OracleQuantileController"]


class OracleQuantileController(Controller):
    """Sets deadlines at the TRUE completion-time quantiles (latent oracle)."""

    def __init__(
        self,
        num_tiers: int,
        targets: tuple[float, ...],
        mu: np.ndarray,
        availability: np.ndarray,
        latent_config: LatentConfig,
        seed: int = 0,
        probe_draws: int = 8,
        min_spacing: float = 0.05,
    ):
        """
        targets: target cumulative fractions q_1<...<q_K (last typically 1.0).
        mu, availability, latent_config: the latent population parameters (ground
            truth) used to estimate the true completion-time CDF.
        """
        super().__init__(num_tiers)
        self._q = np.array(targets, dtype=np.float64)
        self._min_spacing = min_spacing

        # Estimate the TRUE completion-time distribution by sampling the latent
        # model directly (this is the oracle's unfair knowledge).
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(probe_draws):
            mask = rng.random(mu.shape[0]) < availability
            ids = np.flatnonzero(mask)
            samples.append(draw_completion_times(mu[ids], latent_config, rng))
        all_tau = np.concatenate(samples)

        # Deadlines at the true quantiles. For q_k < 1 use the quantile; for the
        # final tier (q_K ~ 1) use a covering cutoff.
        cuts = []
        for qk in self._q:
            if qk >= 1.0:
                cuts.append(float(all_tau.max() * 1.2))
            else:
                cuts.append(float(np.quantile(all_tau, qk)))
        self._cutoffs = tuple(
            project_monotone(np.array(cuts), min_spacing, 0.0, float(all_tau.max() * 3))
        )

    def next_deadlines(self, round_index: int, view: TranscriptView) -> tuple[float, ...]:
        # Oracle is stationary: it already knows the true CDF, so deadlines are
        # fixed at the true quantiles. (A drifting oracle could re-probe; we keep
        # it stationary as the clean upper-bound reference.)
        return self._cutoffs
