"""Send-delay jitter.

Each device delays its upload by a random amount within the tier's allowed
window before the masked-upload cutoff. This decorrelates the FINE release timing
from the device's true capability: a fast device that jitters late looks, on the
timing channel, like a slower one. The attacker's timing-bucket feature is thus
blurred even when buckets are fine.

Privacy effect: adds noise the attacker cannot remove to the release-time signal
(adversary spec Section 5). Utility cost: a device that jitters past the cutoff
becomes a missed-deadline dropout, so aggressive jitter raises dropout / lowers
participation.

We model jitter as an additive delay (in completion-time units) drawn per device
per round, scaled by ``send_delay_scale`` times the tier window width. A device
whose jittered completion time exceeds its tier deadline drops to the next tier
or misses entirely -- handled by re-running tier assignment on the jittered time.
"""

from __future__ import annotations

import numpy as np

from dtfl.defense.config import DefenseConfig

__all__ = ["apply_send_delay"]


def apply_send_delay(
    tau: np.ndarray,
    deadlines_cutoffs: np.ndarray,
    config: DefenseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add within-window jitter to completion times.

    Parameters
    ----------
    tau:
        Completion times for participating devices, shape (n,).
    deadlines_cutoffs:
        The round's tier cutoffs (used to size the jitter window).
    config:
        ``send_delay_scale`` (0 disables) sets jitter magnitude as a fraction of
        the median inter-tier gap.

    Returns
    -------
    Jittered completion times (>= original; jitter only delays, never speeds up,
    since a device cannot finish earlier than it computes). Devices may thereby
    fall into a later tier or miss the round -- the caller re-runs tier
    assignment on the returned times.
    """
    tau = np.asarray(tau, dtype=np.float64)
    if config.send_delay_scale <= 0.0 or tau.size == 0:
        return tau.copy()

    cutoffs = np.asarray(deadlines_cutoffs, dtype=np.float64)
    # Window scale: a fraction of the typical inter-tier gap.
    if cutoffs.size >= 2:
        gap = float(np.median(np.diff(cutoffs)))
    else:
        gap = float(cutoffs[0])
    scale = config.send_delay_scale * gap

    # Half-normal delay (nonnegative): jitter only pushes later.
    jitter = np.abs(rng.normal(0.0, scale, size=tau.size))
    return tau + jitter
