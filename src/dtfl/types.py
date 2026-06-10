"""Shared types for the simulator.

This module deliberately separates two kinds of state by *type*, mirroring the
threat-model boundary from the adversary spec:

- ``LatentDeviceState`` / latent draws: ground truth. Only the simulator's
  ``latent`` and ``protocol`` layers touch these. The ``attack`` package must
  never receive an object of these types.
- ``TierRecord``: the legal observation set. This is the ONLY thing the
  adversary is allowed to see. Its fields are exactly the whitelist from the
  adversary spec.

Keeping these as distinct types lets the separation tests check the boundary
structurally: any function in ``attack`` that accepted a ``LatentDeviceState``
would be a visible type error, not a silent leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = [
    "CapabilityClass",
    "TierFlag",
    "LatentDeviceState",
    "TierRecord",
    "RoundDeadlines",
]


class TierFlag(str, Enum):
    """Outcome flag for a tier session in a round (visible to the server)."""

    RELEASED = "released"
    SUPPRESSED = "suppressed"  # below m_min, or reconstruction failed


# Capability classes are 1..C with 1 = fastest/highest-capability.
# We use a plain int alias rather than an Enum because C is configurable.
CapabilityClass = int


@dataclass(frozen=True)
class LatentDeviceState:
    """Ground-truth, persistent per-device state. NEVER visible to the attacker.

    ``mu`` is the per-device base log-latency (class mean + per-device random
    effect); it is the persistent "fingerprint" that the linkability attack
    would exploit if it could see it. It cannot.
    """

    device_id: int
    capability_class: CapabilityClass  # theta_i, the attack target (Goal A)
    mu: float  # base log-latency: m_c + u_i
    availability_rate: float  # P(participates in a given round)


@dataclass(frozen=True)
class RoundDeadlines:
    """The deadline vector the server broadcasts at the start of a round.

    Public by construction (the server chose it), so it is allowed in the
    transcript. ``cutoffs`` is strictly increasing; ``cutoffs[k]`` is the
    masked-upload deadline for tier k (preamble: deadline == D_mask).
    """

    round_index: int
    cutoffs: tuple[float, ...]  # length K, strictly increasing

    @property
    def num_tiers(self) -> int:
        return len(self.cutoffs)


@dataclass(frozen=True)
class TierRecord:
    """THE LEGAL OBSERVATION SET — the only thing the adversary may see.

    One record per (tier k, round r) that the server processes. Fields match
    the adversary-spec whitelist exactly. If a field is not here, the attacker
    must not have it. Adding a field here widens the attacker's view and MUST be
    a conscious threat-model change reflected in the spec and the whitelist test.
    """

    round_index: int
    tier_index: int
    flag: TierFlag
    # Active-set count (|A_{k,r}|). None only if a defense hides counts entirely.
    count: int | None
    # Coarse release-time bucket (NOT a raw timestamp). None if suppressed before timing.
    release_bucket: int | None
    # Secure sum over the active set. In Steps 0-2 this is a synthetic vector;
    # at Step 4 it is the real aggregated model update. The attacker treats it
    # as opaque metadata (it operates on count/timing/tier, not S contents).
    secure_sum: np.ndarray | None
    # The public deadline vector for this round (server set it; public).
    deadlines: RoundDeadlines

    def visible_field_names(self) -> frozenset[str]:
        """The set of field names a conforming attacker is allowed to read.

        Used by the positive-whitelist separation test. ``secure_sum`` is
        included because it is legitimately released, but note the attacker is
        expected to treat it as opaque.
        """
        return frozenset(
            {
                "round_index",
                "tier_index",
                "flag",
                "count",
                "release_bucket",
                "secure_sum",
                "deadlines",
            }
        )
