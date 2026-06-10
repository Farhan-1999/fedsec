"""Go/no-go gates (adversary spec Section 8).

These determine, early and cheaply, whether the attack-based paper exists. They
run on the minimal simulator + L1 attacker, BEFORE the controller / crypto-cost /
RL layers are built.

  Gate 1: observability-separation test passes (enforced elsewhere, by the
          import-graph and whitelist tests; referenced here for completeness).
  Gate 2: leakage-is-nonzero -- with NO defenses, the L1 attacker's capability
          advantage is meaningfully > 0. If not, there is no defense story;
          pivot to a measurement/negative-result paper consciously.
  Gate 3: defenses-are-not-free -- the strongest defense config measurably
          reduces the advantage. If not, the mechanism contributes nothing.
  Gate 4: defenses-are-not-trivial -- the strongest defense does NOT drive the
          advantage to ~0 at zero utility cost. If it does, suspect a leak in
          the threat model or a too-weak attacker; audit before trusting.

Gates 2-4 take an ``advantage_fn`` so this module has no dependency on the attack
package (which keeps the import graph clean -- sim must not import attack). The
experiment script wires the L1 attacker's advantage in.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dtfl.latent.population import class_tier_mutual_information
from dtfl.sim.engine import SimulationOutput

__all__ = ["GateReport", "snr_diagnostic", "evaluate_gates"]

# advantage_fn: given a SimulationOutput (transcript + ground truth for eval),
# return the L1 capability-inference advantage in [0, 1]. Supplied by the
# experiment script so this module never imports dtfl.attack.
AdvantageFn = Callable[[SimulationOutput], float]


@dataclass
class GateReport:
    """Outcome of the gate battery. ``paper_exists`` is the headline."""

    snr: float
    class_tier_mi: float  # ground-truth leakage diagnostic (nats)
    adv_undefended: float
    adv_defended: float
    utility_cost_of_defense: float
    gate2_leakage_nonzero: bool
    gate3_defense_reduces: bool
    gate4_defense_not_trivial: bool

    @property
    def paper_exists(self) -> bool:
        """All three attack-dependent gates pass."""
        return (
            self.gate2_leakage_nonzero
            and self.gate3_defense_reduces
            and self.gate4_defense_not_trivial
        )

    def summary(self) -> str:
        lines = [
            f"SNR (latent)              : {self.snr:.3f}",
            f"class->tier MI (nats)     : {self.class_tier_mi:.4f}",
            f"L1 advantage  undefended  : {self.adv_undefended:.4f}",
            f"L1 advantage  defended    : {self.adv_defended:.4f}",
            f"utility cost of defense   : {self.utility_cost_of_defense:.4f}",
            f"Gate 2 leakage nonzero    : {'PASS' if self.gate2_leakage_nonzero else 'FAIL'}",
            f"Gate 3 defense reduces    : {'PASS' if self.gate3_defense_reduces else 'FAIL'}",
            f"Gate 4 defense not trivial: {'PASS' if self.gate4_defense_not_trivial else 'FAIL'}",
            f"==> PAPER EXISTS          : {'YES' if self.paper_exists else 'NO (pivot)'}",
        ]
        return "\n".join(lines)


def snr_diagnostic(output: SimulationOutput) -> tuple[float, float]:
    """Ground-truth leakage diagnostic from a simulation output.

    Aggregates the per-round latent logs into (true class, assigned tier) pairs
    over all active participations and computes I(class; tier). This uses latent
    labels and is EXPERIMENTER-ONLY -- it never touches the transcript view.

    Returns (placeholder_snr, mutual_information). SNR itself is a config-level
    quantity; we return MI as the empirical realized leakage and leave SNR to the
    caller (it has the LatentConfig). Here we return (mi, mi) when SNR unknown.
    """
    classes_all = []
    tiers_all = []
    for log in output.latent_logs:
        for k, cls in enumerate(log.active_classes):
            if cls.size:
                classes_all.append(cls)
                tiers_all.append(np.full(cls.size, k, dtype=np.int64))
    if not classes_all:
        return 0.0, 0.0
    classes_cat = np.concatenate(classes_all)
    tiers_cat = np.concatenate(tiers_all)
    mi = class_tier_mutual_information(classes_cat, tiers_cat)
    return mi, mi


def evaluate_gates(
    undefended_output: SimulationOutput,
    defended_output: SimulationOutput,
    advantage_fn: AdvantageFn,
    snr: float,
    *,
    leakage_threshold: float = 0.05,
    reduction_threshold: float = 0.02,
    triviality_floor: float = 0.02,
    free_defense_cost_eps: float = 1e-6,
) -> GateReport:
    """Run gates 2-4 given an undefended and a strongest-defended run.

    Parameters
    ----------
    advantage_fn:
        Computes L1 capability-inference advantage from a SimulationOutput.
    snr:
        The latent SNR of the run (from LatentConfig.signal_to_noise), reported
        for context.
    leakage_threshold:
        Gate 2 passes if undefended advantage exceeds this.
    reduction_threshold:
        Gate 3 passes if (undefended - defended) advantage exceeds this.
    triviality_floor / free_defense_cost_eps:
        Gate 4 FAILS (suspicious) only if the defense drives advantage below the
        floor AND costs essentially nothing in utility.
    """
    adv_undef = advantage_fn(undefended_output)
    adv_def = advantage_fn(defended_output)

    # Utility cost: drop in mean participation (proxy for time-to-accuracy cost)
    # from applying the defense. More defense -> more suppression -> less
    # participation. A real utility cost makes a near-zero advantage legitimate.
    util_undef = float(np.mean(undefended_output.participation))
    util_def = float(np.mean(defended_output.participation))
    utility_cost = max(0.0, util_undef - util_def) / max(1.0, util_undef)

    _, mi = snr_diagnostic(undefended_output)

    gate2 = adv_undef > leakage_threshold
    gate3 = (adv_undef - adv_def) > reduction_threshold
    # Gate 4 fails only if privacy came for free: advantage crushed AND no cost.
    defense_is_free = (adv_def < triviality_floor) and (utility_cost <= free_defense_cost_eps)
    gate4 = not defense_is_free

    return GateReport(
        snr=snr,
        class_tier_mi=mi,
        adv_undefended=adv_undef,
        adv_defended=adv_def,
        utility_cost_of_defense=utility_cost,
        gate2_leakage_nonzero=gate2,
        gate3_defense_reduces=gate3,
        gate4_defense_not_trivial=gate4,
    )
