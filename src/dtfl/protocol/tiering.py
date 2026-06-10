"""Tier self-assignment.

Devices self-assign to the EARLIEST tier whose masked-upload deadline they
expect to meet (preamble: the broadcast deadline IS the masked-upload cutoff
D_mask, and tau already includes the SecAgg setup+upload phases). A device that
meets no deadline is a missed-deadline dropout for the round and contributes to
no tier.

This is the only place the latent completion time tau touches tier structure.
The output here is a per-device tier index that the protocol layer then turns
into rosters; the index itself is latent (it is per-device) and never reaches
the transcript -- only aggregate counts and the released sum do.
"""

from __future__ import annotations

import numpy as np

from dtfl.types import RoundDeadlines

__all__ = ["assign_tiers", "MISSED"]

# Sentinel tier index for devices that miss every deadline (round dropout).
MISSED = -1


def assign_tiers(tau: np.ndarray, deadlines: RoundDeadlines) -> np.ndarray:
    """Assign each device to the earliest tier whose deadline it meets.

    Parameters
    ----------
    tau:
        Per-device completion times, shape (n,). Already includes SecAgg phases.
    deadlines:
        The round's deadline vector; ``cutoffs`` strictly increasing, length K.

    Returns
    -------
    Array of tier indices in 0..K-1, or ``MISSED`` (-1) for round dropouts,
    shape (n,). A device with tau <= cutoffs[k] for the smallest such k is
    assigned to tier k (earliest feasible tier).
    """
    tau = np.asarray(tau, dtype=np.float64)
    cutoffs = np.asarray(deadlines.cutoffs, dtype=np.float64)
    if cutoffs.size == 0:
        raise ValueError("deadline vector is empty")
    if np.any(np.diff(cutoffs) <= 0):
        raise ValueError(f"deadline cutoffs must be strictly increasing, got {cutoffs}")

    # searchsorted finds, for each tau, the index of the first cutoff >= tau.
    # side='left' so tau exactly equal to a cutoff counts as meeting it.
    tiers = np.searchsorted(cutoffs, tau, side="left").astype(np.int64)
    # tau strictly greater than the last cutoff -> index K -> a missed deadline.
    tiers[tiers >= cutoffs.size] = MISSED
    return tiers


def tier_rosters(tiers: np.ndarray, num_tiers: int) -> list[np.ndarray]:
    """Group device indices by assigned tier.

    Returns a list of length ``num_tiers``; entry k is the array of device
    indices assigned to tier k. Missed-deadline devices are excluded entirely.
    """
    tiers = np.asarray(tiers)
    return [np.flatnonzero(tiers == k) for k in range(num_tiers)]
