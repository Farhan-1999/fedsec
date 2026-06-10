"""Anonymity-floor adapter.

``m_min`` is enforced in the release gate (protocol/release.py); this module just
surfaces the floor and the tier count from a DefenseConfig, plus a helper to
build the deadline quantiles for the configured number of tiers. More tiers at a
fixed m_min means smaller per-tier mass and thus more suppression -- so K and
m_min together form the suppression axis.
"""

from __future__ import annotations

from dtfl.defense.config import DefenseConfig

__all__ = ["deadline_quantiles_for", "m_min_of"]


def m_min_of(config: DefenseConfig) -> int:
    return config.m_min


def deadline_quantiles_for(config: DefenseConfig) -> tuple[float, ...]:
    """Evenly spaced interior quantiles for ``num_tiers`` tiers.

    Returns the K-1 interior cut quantiles; the engine's calibration appends a
    covering final tier. For K tiers we cut at 1/K, 2/K, ..., (K-1)/K.
    """
    K = config.num_tiers
    if K < 1:
        raise ValueError("num_tiers must be >= 1")
    if K == 1:
        return ()
    return tuple(i / K for i in range(1, K))
