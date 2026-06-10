"""Reconstruction-threshold selection (authoritative rule, preamble Section 6).

This resolves the cross-document threshold conflict. The authoritative rule is
the dropout doc's Hoeffding-derived threshold, clamped to t <= m_min:

    mu_k = n_k * (1 - rho_M) * (1 - rho_U)            # expected unmask survivors
    t_k  = min( m_min, floor( mu_k - sqrt( (n_k / 2) * ln(1/delta) ) ) )

The clamp enforces the SAFE-COUPLING INVARIANT: if a tier is allowed to release
(meets m_min), it must also be reconstructable (meets t). The crypto doc's
competing formula t = ceil((1-delta) * n_k) is NOT used -- for large tiers it can
exceed m_min and violate the invariant.

t is also floored at 1 (a degenerate but valid threshold) and never exceeds the
number of clients that could possibly respond.
"""

from __future__ import annotations

import math

from dtfl.protocol.dropout import DropoutRates

__all__ = ["reconstruction_threshold", "safe_coupling_holds"]


def reconstruction_threshold(
    n_k: int,
    m_min: int,
    rates: DropoutRates,
    delta: float = 1e-3,
) -> int:
    """Compute the Shamir reconstruction threshold t_k for a tier.

    Parameters
    ----------
    n_k:
        Active-set size (|A_{k,r}|) -- the count that will be revealed and used
        as the merge weight.
    m_min:
        Anonymity floor; a tier only releases if n_k >= m_min.
    rates:
        Dropout rates; rho_mask and rho_unmask give the expected unmask survival.
    delta:
        Target failure probability for the Hoeffding tail bound.

    Returns
    -------
    Integer threshold t_k in [1, m_min]. Guaranteed t_k <= m_min (the invariant).
    """
    if n_k <= 0:
        return 1
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0,1), got {delta}")

    # Expected number surviving to the unmask phase.
    mu_k = n_k * (1.0 - rates.rho_mask) * (1.0 - rates.rho_unmask)
    # Hoeffding slack for a sum of n_k bounded (Bernoulli) variables.
    slack = math.sqrt((n_k / 2.0) * math.log(1.0 / delta))
    hoeffding_t = math.floor(mu_k - slack)

    # Clamp into [1, m_min]; the m_min clamp is the safe-coupling invariant.
    t_k = min(m_min, hoeffding_t)
    t_k = max(1, t_k)
    return int(t_k)


def safe_coupling_holds(t_k: int, m_min: int) -> bool:
    """The invariant: a releasable tier (n_k >= m_min) is always reconstructable.

    Since release requires n_k >= m_min and we require t_k <= m_min, any released
    tier with all m_min-plus clients responding can clear the threshold. Used by
    tests to assert the rule never produces t_k > m_min.
    """
    return 1 <= t_k <= m_min
