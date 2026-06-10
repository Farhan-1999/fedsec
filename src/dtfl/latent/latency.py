"""Completion-time draws.

Given a device's persistent base log-latency ``mu_i`` and the config, draw the
round-specific observed completion time

    log tau_{i,r} = mu_i + eps_{i,r} + tail_{i,r} [+ shared regime term]
    tau_{i,r}     = exp(log tau_{i,r})

where ``eps ~ Normal(0, eta)`` is the proxy noise (the privacy-protecting blur)
and ``tail`` is, with probability ``tail_prob``, an ``Exponential(tail_scale)``
spike (straggler bursts / misclassification events).

We also expose the four-term decomposition (train / setup / upload / jitter) so
the secure-aggregation cost model (Step 6) can later inflate the setup+upload
terms by tier size and graph topology. In Steps 0-2 the split is purely notional:
only the total ``tau`` drives tiering, and only the resulting tier + coarse
release bucket ever reach the transcript. The split is latent.

All functions are vectorized over devices: pass arrays of ``mu`` and get arrays
of ``tau`` back, since the engine draws for all participating devices per round.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dtfl.latent.config import LatentConfig

__all__ = ["draw_completion_times", "LatencyDecomposition", "decompose"]


def draw_completion_times(
    mu: np.ndarray,
    config: LatentConfig,
    rng: np.random.Generator,
    shared_log_offset: float = 0.0,
) -> np.ndarray:
    """Draw observed completion times ``tau`` for devices with base log-latency ``mu``.

    Parameters
    ----------
    mu:
        Array of per-device base log-latencies (``m_c + u_i``), shape (n,).
    config:
        Latent model parameters; ``proxy_noise_eta``, ``tail_prob``,
        ``tail_scale_log`` are used here.
    rng:
        Generator for this round's draws.
    shared_log_offset:
        A round-level additive term on log tau, shared across all devices, used
        by the drift layer to inject regime shifts / slow drift. 0 in the clean
        gate setting.

    Returns
    -------
    tau:
        Array of positive completion times, shape (n,). NaN-free, finite.
    """
    mu = np.asarray(mu, dtype=np.float64)
    n = mu.shape[0]

    # Proxy noise: the central privacy knob.
    eps = rng.normal(0.0, config.proxy_noise_eta, size=n)

    # Heavy tail: Bernoulli gate times an Exponential spike (additive on log).
    tail = np.zeros(n, dtype=np.float64)
    if config.tail_prob > 0.0 and config.tail_scale_log > 0.0:
        hit = rng.random(n) < config.tail_prob
        spikes = rng.exponential(config.tail_scale_log, size=n)
        tail = np.where(hit, spikes, 0.0)

    log_tau = mu + eps + tail + shared_log_offset
    tau = np.exp(log_tau)
    # Guard against pathological overflow from an extreme tail draw.
    if not np.all(np.isfinite(tau)):
        tau = np.nan_to_num(tau, posinf=np.finfo(np.float64).max, nan=np.finfo(np.float64).max)
    return tau


@dataclass(frozen=True)
class LatencyDecomposition:
    """Latent four-term split of a completion time. NEVER reaches the transcript.

    Fractions are notional in Steps 0-2 (fixed split of the total); they exist so
    the Step-6 crypto cost model can replace ``setup`` and ``upload`` with
    tier-size- and topology-dependent values without restructuring the engine.
    """

    train: np.ndarray
    setup: np.ndarray
    upload: np.ndarray
    jitter: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.train + self.setup + self.upload + self.jitter


# Notional split used until the crypto cost model overrides it (Step 6).
# train dominates; setup+upload are the SecAgg-phase share; jitter is small.
_DEFAULT_SPLIT = (0.70, 0.10, 0.15, 0.05)  # train, setup, upload, jitter


def decompose(tau: np.ndarray, split: tuple[float, float, float, float] = _DEFAULT_SPLIT) -> LatencyDecomposition:
    """Split total completion times into the four notional phase components.

    A deterministic proportional split for now. Step 6 swaps the setup/upload
    portions for cost-model outputs that depend on tier size and graph topology.
    """
    if abs(sum(split) - 1.0) > 1e-9:
        raise ValueError(f"split must sum to 1, got {sum(split)}")
    tau = np.asarray(tau, dtype=np.float64)
    tr, se, up, ji = split
    return LatencyDecomposition(
        train=tau * tr,
        setup=tau * se,
        upload=tau * up,
        jitter=tau * ji,
    )
