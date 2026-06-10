"""Nonstationarity / drift.

A stateful object the engine steps once per round. It produces:
- a ``shared_log_offset`` (regime shifts + slow global drift) passed into the
  latency draw and applied equally to all devices that round, and
- an in-place-ish per-device random walk on ``mu`` (returned as a new array)
  that makes each device's fingerprint persistent-but-not-eternal, bounding how
  long the linkability attack can track a device and shaping the L_i(h)-vs-h
  curve.

All draws come from a dedicated stream so toggling drift on/off does not perturb
the latency or population streams (reproducibility property from rng.py).

With ``DriftConfig`` all-zero (the gate default) this is a no-op: offset stays 0
and mu is returned unchanged, so the clean-signal gate run is unaffected.
"""

from __future__ import annotations

import numpy as np

from dtfl.latent.config import DriftConfig

__all__ = ["DriftState"]


class DriftState:
    """Evolving global level and per-device walk. Stepped once per round.

    Usage:
        drift = DriftState(config.drift, n_devices=N, rng=hub.stream("latent.drift"))
        for r in range(R):
            offset = drift.step()          # advance global level; get this round's offset
            mu_r = drift.apply_device_walk(mu_r)   # advance per-device fingerprints
            tau = draw_completion_times(mu_r, config, lat_rng, shared_log_offset=offset)
    """

    def __init__(self, config: DriftConfig, n_devices: int, rng: np.random.Generator):
        self._cfg = config
        self._n = n_devices
        self._rng = rng
        # Current global log-level (sum of accumulated regime shifts + slow drift).
        self._global_level = 0.0

    @property
    def global_level(self) -> float:
        return self._global_level

    def step(self) -> float:
        """Advance the global level by one round and return the offset to apply.

        Regime shift: with prob ``regime_shift_prob`` add a one-off level jump of
        stddev ``regime_shift_scale``. Slow drift: always add a small random-walk
        increment of stddev ``slow_drift_per_round``. Both accumulate into the
        persistent global level (a regime shift is a lasting change, not a
        one-round blip), which is the realistic model of a network entering a
        congested period.
        """
        if not self._cfg.enabled:
            return 0.0
        if self._cfg.regime_shift_prob > 0.0 and self._rng.random() < self._cfg.regime_shift_prob:
            self._global_level += self._rng.normal(0.0, self._cfg.regime_shift_scale)
        if self._cfg.slow_drift_per_round > 0.0:
            self._global_level += self._rng.normal(0.0, self._cfg.slow_drift_per_round)
        return self._global_level

    def apply_device_walk(self, mu: np.ndarray) -> np.ndarray:
        """Apply one step of the per-device random walk on base log-latency.

        Returns a new array (does not mutate the input). No-op if the walk scale
        is zero. The walk is persistent: each round's increment accumulates, so a
        device slowly drifts away from its original fingerprint over many rounds.
        """
        mu = np.asarray(mu, dtype=np.float64)
        if self._cfg.device_walk_per_round <= 0.0:
            return mu.copy()
        increments = self._rng.normal(0.0, self._cfg.device_walk_per_round, size=mu.shape[0])
        return mu + increments
