"""Merging tier aggregates into one global update (preamble Section 7).

Under IID with revealed active counts, the per-round merge is the size-weighted
mean over successful tiers, which equals FedAvg over the union of participating
clients:

    Delta_bar_r = (1 / n_r) * sum_{k in K_r} S_{k,r},   n_r = sum_{k in K_r} n_{k,r}
    w_{r+1}     = w_r + eta_r * Delta_bar_r

The merge weight is the ACTIVE count n_{k,r}. This couples privacy to utility:
if a count-hiding/noising defense perturbs n_{k,r}, the weight is wrong and the
merged update is biased -- that bias is a utility cost we measure
(metrics/merge_error.py), not something we paper over.

Synchronous per-round merge is the mainline. Buffered/staleness-weighted merge
is the studied variant (separate function). Momentum and norm-clipping are
optional server-side refinements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TierContribution",
    "size_weighted_merge",
    "weighted_merge",
    "apply_server_update",
]


@dataclass(frozen=True)
class TierContribution:
    """One successful tier's contribution to a merge.

    ``secure_sum`` is S_{k,r} (sum of updates over the active set); ``count`` is
    n_{k,r} (the weight). ``staleness`` is the application delay used only by the
    buffered/staleness-weighted variant.
    """

    secure_sum: np.ndarray
    count: int
    staleness: float = 0.0


def size_weighted_merge(contributions: list[TierContribution]) -> np.ndarray | None:
    """FedAvg-equivalent merged mean update over successful tiers.

    Returns ``(1/n_r) * sum_k S_{k,r}`` where ``n_r = sum_k n_{k,r}``, or None if
    there is no participation (no successful tiers / total count 0), in which case
    the caller leaves the model unchanged.

    Equivalence: since tiers partition the participating clients, summing the tier
    sums and dividing by the total count recovers exactly the FedAvg average over
    the union of participants -- this is the merging theorem (preamble Section 7).
    """
    if not contributions:
        return None
    total = sum(c.count for c in contributions)
    if total <= 0:
        return None
    dim = contributions[0].secure_sum.shape
    acc = np.zeros(dim, dtype=np.float64)
    for c in contributions:
        acc += np.asarray(c.secure_sum, dtype=np.float64)
    return acc / total


def weighted_merge(
    contributions: list[TierContribution],
    staleness_lambda: float = 0.0,
) -> np.ndarray | None:
    """General weighted merge with optional staleness down-weighting.

    Weight for tier k: alpha_k proportional to n_{k,r} * exp(-lambda * s_{k,r}),
    normalized to sum to 1, applied to the per-tier MEAN (S_{k,r}/n_{k,r}).

    With ``staleness_lambda == 0`` this reduces exactly to size weighting and
    therefore to ``size_weighted_merge`` (up to floating point). Used for the
    buffered/asynchronous variant where tiers arrive with different delays.
    """
    if not contributions:
        return None
    weights = np.array(
        [c.count * np.exp(-staleness_lambda * c.staleness) for c in contributions],
        dtype=np.float64,
    )
    wsum = weights.sum()
    if wsum <= 0:
        return None
    weights /= wsum
    dim = contributions[0].secure_sum.shape
    acc = np.zeros(dim, dtype=np.float64)
    for w, c in zip(weights, contributions, strict=True):
        mean_k = np.asarray(c.secure_sum, dtype=np.float64) / max(1, c.count)
        acc += w * mean_k
    return acc


def apply_server_update(
    w: np.ndarray,
    delta: np.ndarray | None,
    lr: float,
    momentum: float = 0.0,
    velocity: np.ndarray | None = None,
    clip_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply a merged update to the global model with optional momentum/clipping.

    Returns ``(w_next, velocity_next)``. If ``delta`` is None (no participation),
    the model is returned unchanged and velocity is preserved.

    Clipping is applied to the merged update norm before the LR/momentum step,
    which stabilizes training under bursty tiers.
    """
    if delta is None:
        return w, velocity

    delta = np.asarray(delta, dtype=np.float64)
    if clip_norm is not None:
        norm = float(np.linalg.norm(delta))
        if norm > clip_norm and norm > 0:
            delta = delta * (clip_norm / norm)

    if momentum > 0.0:
        v_prev = velocity if velocity is not None else np.zeros_like(delta)
        velocity_next = momentum * v_prev + delta
        w_next = w + lr * velocity_next
        return w_next, velocity_next

    return w + lr * delta, velocity
