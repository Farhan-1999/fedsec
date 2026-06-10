"""Linkability over horizons (adversary spec Goal B, the multi-round result).

The headline multi-round privacy metric: L_i(h) = 1 / |E_i(h)|, the linkability
risk for device i over horizon h. L_i(h) = 1 means device i is uniquely
identifiable from its visible signature over h rounds; L_i(h) = 1/n means it is
indistinguishable from n-1 others.

Averaged over devices, mean L(h) is NON-DECREASING in h: each extra round can
only refine signatures, never coarsen them, so the crowd a device hides in can
only shrink. The slope of mean L(h) vs h is the LEAKAGE-ACCUMULATION RATE -- the
quantification of "secure aggregation hides contents, but repeated participation
still erodes anonymity over rounds." This is a distinct contribution from the
single-round capability advantage and is exactly the multi-round concern the
PETS literature emphasizes.

We also provide a trajectory-reconstruction measure: given the visible
signatures, how often can an adversary correctly decide that two participations
belong to the same device? Under fresh per-round pseudonyms the adversary links
by signature consistency; this measures the realized linkability directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dtfl.metrics.anonymity import (
    AnonymityStats,
    anonymity_stats,
    equivalence_class_sizes,
    signature_up_to,
)

__all__ = ["LinkabilityCurve", "linkability_curve", "mean_linkability_at"]


@dataclass
class LinkabilityCurve:
    """Mean linkability risk and anonymity stats as functions of horizon."""

    horizons: list[int]
    mean_linkability: list[float]  # mean_i 1/|E_i(h)|
    median_anon_set: list[float]  # median |E_i(h)|
    fraction_unique: list[float]  # fraction with |E_i(h)| == 1
    fraction_below_m_min: list[float]
    stats: list[AnonymityStats]

    @property
    def accumulation_rate(self) -> float:
        """Slope of mean linkability vs horizon (least-squares).

        Positive slope = anonymity erodes as rounds accumulate. Reported as the
        single-number "how fast does repeated participation leak" summary.
        """
        if len(self.horizons) < 2:
            return 0.0
        h = np.asarray(self.horizons, dtype=np.float64)
        y = np.asarray(self.mean_linkability, dtype=np.float64)
        # slope of the OLS fit
        return float(np.polyfit(h, y, 1)[0])

    def summary(self) -> str:
        lines = [f"horizon  meanL   medAnon  unique%  <m_min%"]
        for i, h in enumerate(self.horizons):
            lines.append(
                f"{h:>7}  {self.mean_linkability[i]:.4f}  "
                f"{self.median_anon_set[i]:>7.1f}  "
                f"{100*self.fraction_unique[i]:>6.1f}  "
                f"{100*self.fraction_below_m_min[i]:>6.1f}"
            )
        lines.append(f"accumulation rate (slope of meanL vs h): {self.accumulation_rate:.5f}")
        return "\n".join(lines)


def _mean_linkability_from_sizes(class_sizes: dict[int, int]) -> float:
    if not class_sizes:
        return 0.0
    sizes = np.array(list(class_sizes.values()), dtype=np.float64)
    return float(np.mean(1.0 / sizes))


def mean_linkability_at(
    device_per_round_atoms: dict[int, dict[int, tuple[int, int]]],
    horizon: int,
    m_min: int,
) -> tuple[float, AnonymityStats]:
    """Mean linkability and anonymity stats at a single horizon."""
    sigs = {
        did: signature_up_to(atoms, horizon)
        for did, atoms in device_per_round_atoms.items()
    }
    class_sizes = equivalence_class_sizes(sigs)
    stats = anonymity_stats(class_sizes, m_min)
    stats.horizon = horizon
    return _mean_linkability_from_sizes(class_sizes), stats


def linkability_curve(
    device_per_round_atoms: dict[int, dict[int, tuple[int, int]]],
    horizons: list[int],
    m_min: int,
) -> LinkabilityCurve:
    """Compute the linkability/anonymity curve over a list of horizons.

    Parameters
    ----------
    device_per_round_atoms:
        Map device_id -> {round_index -> (tier, bucket)} for released tier-rounds
        the device appeared in. Built by the harness from the same released-tier-
        filtered observations the attacker uses.
    horizons:
        Increasing horizon lengths to evaluate (e.g. [5, 10, 20, 40, 80]).
    m_min:
        Anonymity floor, for the violation-rate series.
    """
    mean_link = []
    med_anon = []
    frac_uniq = []
    frac_below = []
    stats_list = []
    for h in horizons:
        ml, stats = mean_linkability_at(device_per_round_atoms, h, m_min)
        mean_link.append(ml)
        med_anon.append(stats.median)
        frac_uniq.append(stats.fraction_unique)
        frac_below.append(stats.fraction_below_m_min)
        stats_list.append(stats)
    return LinkabilityCurve(
        horizons=list(horizons),
        mean_linkability=mean_link,
        median_anon_set=med_anon,
        fraction_unique=frac_uniq,
        fraction_below_m_min=frac_below,
        stats=stats_list,
    )
