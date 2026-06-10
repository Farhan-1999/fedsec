"""Padding: inject dummy (zero-update) contributions to inflate small tiers.

A tier whose active count is below the padding target gets dummy masked
contributions added so its visible count reaches the target. In real SecAgg these
are masked zero-updates from helper identities; here we model the effect on the
transcript (the revealed count rises) and on the merge (dummy zero-updates do not
change the sum but DO change the weight, so the merge mean is diluted toward
zero -- a utility cost).

Privacy effect: raises the visible count and the realized anonymity set for small
tiers, so a tier that would otherwise be suppressed (or be a small, identifying
anonymity set) is inflated (adversary spec Section 5). Utility cost: dummy
zero-updates dilute the merged mean (the sum is unchanged but divided by a larger
count), and the padding itself costs compute/comm.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PaddingResult", "apply_padding"]


@dataclass(frozen=True)
class PaddingResult:
    """Effect of padding on one tier."""

    revealed_count: int  # count shown in the transcript (>= true active count)
    num_dummies: int  # how many dummy zero-updates were added
    # The merge sum is unchanged (dummies are zero); the merge weight becomes
    # ``revealed_count``, so the merged mean for this tier is scaled by
    # true_count / revealed_count -- the dilution the experiment can measure.


def apply_padding(true_active_count: int, padding_target: int) -> PaddingResult:
    """Pad a tier's count up toward ``padding_target`` with dummy zero-updates.

    No-op if padding is disabled (target 0) or the tier already meets the target.
    """
    if padding_target <= 0 or true_active_count >= padding_target:
        return PaddingResult(revealed_count=true_active_count, num_dummies=0)
    dummies = padding_target - true_active_count
    return PaddingResult(revealed_count=padding_target, num_dummies=dummies)
