"""Phase-specific dropout (structured dropout model from the dropout doc).

A roster (devices that joined a tier) is winnowed by independent per-phase
survival into the ACTIVE SET (devices whose masked upload is included). The two
relevant dropout rates:

- ``rho_mask``: probability a rostered client fails to deliver its masked upload
  by D_mask + grace (network failure, mid-protocol crash, or simply slow). This
  is the dominant attrition and the one the missed-deadline structure feeds.
- ``rho_unmask``: probability an active client fails to respond in the unmask
  phase (matters for reconstruction feasibility, not for who is in the sum).

The active count n_{k,r} = |A_{k,r}| is what becomes the (revealed) transcript
count and the merge weight. Whether the tier can be UNMASKED depends on enough
clients surviving the unmask phase to meet the threshold t (see threshold.py and
release.py); that is a separate gate from who is in A.

This module is deliberately a pure stochastic model: it does not run a real
protocol. It takes a roster size (or roster indices) and returns survival masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DropoutRates", "apply_dropout", "DropoutOutcome"]


@dataclass(frozen=True)
class DropoutRates:
    """Per-phase dropout probabilities. Estimated from aggregate counts via EWMA
    in the running system; fixed parameters in the simulator."""

    rho_mask: float = 0.10  # fail to deliver masked upload
    rho_unmask: float = 0.05  # fail to respond in unmask phase (given active)

    def __post_init__(self) -> None:
        for name, v in (("rho_mask", self.rho_mask), ("rho_unmask", self.rho_unmask)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {v}")


@dataclass(frozen=True)
class DropoutOutcome:
    """Result of winnowing a roster.

    ``active`` are roster-local indices (0..roster_size-1) that delivered a
    masked upload; ``unmask_responders`` are the subset of active that also
    responded in the unmask phase. Sizes:
        roster_size >= |active| >= |unmask_responders|.
    """

    roster_size: int
    active: np.ndarray  # roster-local indices in A
    unmask_responders: np.ndarray  # roster-local indices in A that also unmask

    @property
    def active_count(self) -> int:
        return int(self.active.size)

    @property
    def unmask_count(self) -> int:
        return int(self.unmask_responders.size)


def apply_dropout(
    roster_size: int,
    rates: DropoutRates,
    rng: np.random.Generator,
) -> DropoutOutcome:
    """Winnow a roster of ``roster_size`` into active and unmask-responding sets.

    Two independent Bernoulli passes:
      1. masked-upload survival with prob (1 - rho_mask) -> active set A,
      2. unmask survival with prob (1 - rho_unmask) among A -> responders.

    Returns roster-local indices. The caller maps these back to device indices
    (the protocol layer holds the roster); device identities never leave the
    latent/protocol layer.
    """
    if roster_size < 0:
        raise ValueError("roster_size must be >= 0")
    if roster_size == 0:
        empty = np.empty(0, dtype=np.int64)
        return DropoutOutcome(roster_size=0, active=empty, unmask_responders=empty)

    survives_mask = rng.random(roster_size) >= rates.rho_mask
    active = np.flatnonzero(survives_mask)

    if active.size == 0:
        return DropoutOutcome(
            roster_size=roster_size,
            active=active,
            unmask_responders=np.empty(0, dtype=np.int64),
        )

    survives_unmask = rng.random(active.size) >= rates.rho_unmask
    unmask_responders = active[survives_unmask]

    return DropoutOutcome(
        roster_size=roster_size,
        active=active,
        unmask_responders=unmask_responders,
    )
