"""Analytic tier-success probability (dropout doc Section on success probability).

A tier session succeeds (releases) iff enough participants survive to (a) meet the
anonymity floor m_min and (b) clear the reconstruction threshold t. Modeling each
rostered client's survival as independent Bernoulli, the active count and the
unmask-responder count are binomial, giving a closed-form success probability.

This module exists primarily to back the PREDICTED-vs-EMPIRICAL release
calibration plot the experimental design requires: if the analytic prediction
matches the simulator's measured release rate across sweeps of (roster size,
dropout rate, m_min, t), the dropout model is validated and the framework looks
engineered rather than improvised.

Two estimators:
  - exact binomial-tail success probability,
  - a conservative Hoeffding lower bound (distribution-free), matching the
    threshold-selection rule in protocol/threshold.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from dtfl.protocol.dropout import DropoutRates

__all__ = ["SuccessPrediction", "tier_success_probability", "hoeffding_success_lower_bound"]


@dataclass
class SuccessPrediction:
    roster_size: int
    m_min: int
    threshold: int
    p_active: float  # 1 - rho_mask
    p_unmask: float  # 1 - rho_unmask (conditional)
    prob_meets_m_min: float
    prob_meets_threshold: float
    prob_success: float

    def summary(self) -> str:
        return (
            f"n={self.roster_size} m_min={self.m_min} t={self.threshold} "
            f"P(>=m_min)={self.prob_meets_m_min:.3f} "
            f"P(>=t unmask)={self.prob_meets_threshold:.3f} "
            f"P(success)={self.prob_success:.3f}"
        )


def tier_success_probability(
    roster_size: int,
    m_min: int,
    threshold: int,
    rates: DropoutRates,
) -> SuccessPrediction:
    """Exact binomial-tail probability that a tier releases.

    Active count A ~ Binomial(n, 1 - rho_mask). Given A = a, unmask responders
    U ~ Binomial(a, 1 - rho_unmask). The tier succeeds iff A >= m_min AND U >= t.
    Because the release gate checks the ACTIVE count against m_min and the
    UNMASK count against t, we compute:

        P(success) = sum_{a >= m_min} P(A = a) * P(U >= t | A = a).
    """
    n = roster_size
    pA = 1.0 - rates.rho_mask
    pU = 1.0 - rates.rho_unmask
    if n <= 0:
        return SuccessPrediction(n, m_min, threshold, pA, pU, 0.0, 0.0, 0.0)

    # P(A >= m_min)
    prob_m_min = float(stats.binom.sf(m_min - 1, n, pA))  # sf(k-1) = P(X >= k)

    # P(success) = sum_{a=max(m_min,t)}^{n} P(A=a) * P(U>=t | a)
    a_lo = max(m_min, threshold)
    prob_success = 0.0
    prob_thresh_weighted = 0.0
    for a in range(a_lo, n + 1):
        pa = stats.binom.pmf(a, n, pA)
        pu_ge_t = stats.binom.sf(threshold - 1, a, pU)
        prob_success += pa * pu_ge_t
        prob_thresh_weighted += pa * pu_ge_t
    # P(meets threshold | meets m_min) reported descriptively
    prob_thresh = prob_thresh_weighted / prob_m_min if prob_m_min > 0 else 0.0

    return SuccessPrediction(
        roster_size=n, m_min=m_min, threshold=threshold,
        p_active=pA, p_unmask=pU,
        prob_meets_m_min=prob_m_min,
        prob_meets_threshold=min(1.0, prob_thresh),
        prob_success=float(prob_success),
    )


def hoeffding_success_lower_bound(
    roster_size: int,
    threshold: int,
    rates: DropoutRates,
    delta: float = 1e-3,
) -> float:
    """Distribution-free lower bound on P(unmask survivors >= threshold).

    Matches the Hoeffding reasoning behind the threshold-selection rule: with
    expected survivors mu = n*(1-rho_M)*(1-rho_U), the probability the count
    falls more than s below mu is <= exp(-2 s^2 / n). Returns 1 - that bound at
    s = mu - threshold (clamped to [0,1]); a conservative success guarantee.
    """
    n = roster_size
    mu = n * (1 - rates.rho_mask) * (1 - rates.rho_unmask)
    if threshold >= mu:
        return 0.0  # threshold above the mean: no lower-bound guarantee
    s = mu - threshold
    bound = math.exp(-2.0 * s * s / n) if n > 0 else 1.0
    return max(0.0, 1.0 - bound)
