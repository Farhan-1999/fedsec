#!/usr/bin/env python
"""Reproduce all paper results and figures from fixed seeds.

Single entry point for the artifact. Runs each experiment in dependency order and
writes results to artifacts/results/ and figures to artifacts/figures/. Every run
is seeded; re-running reproduces identical outputs (the determinism property is
tested in tests/test_transcript_sim.py).

Usage:
    python reproduce.py [stage]

Stages (default: all):
    gate         - Step 1 go/no-go gate
    pareto       - Step 2 headline privacy-utility frontier + figure
    ladder       - Step 3 adversary ladder L0-L3
    linkability  - linkability/anonymity horizon curve + figure
    multiseed    - multi-seed hardening (frontier + ladder + linkability) + figure
    training     - federated training curves (loss/accuracy/convergence/time)
    secagg       - secure-aggregation calibration + cost comparison
    all          - everything above

Note: figures depend on their results CSVs; running a plot stage assumes the
corresponding results stage has been run (or run 'all').
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"
PLOTS = EXP / "plots"

# (label, script path, is figure?) in dependency order.
STAGES = {
    "gate": [("go/no-go gate", EXP / "run_gate.py")],
    "pareto": [
        ("headline frontier sweep", EXP / "run_headline_pareto.py"),
        ("frontier figure", PLOTS / "pareto.py"),
    ],
    "ladder": [("adversary ladder", EXP / "run_ladder.py")],
    "linkability": [
        ("linkability horizon curve", EXP / "run_linkability.py"),
        ("linkability figure", PLOTS / "linkability_horizon.py"),
    ],
    "multiseed": [
        ("multi-seed frontier", EXP / "run_multiseed_frontier.py"),
        ("multi-seed ladder+linkability", EXP / "run_multiseed_ladder_linkability.py"),
        ("multi-seed frontier figure", PLOTS / "frontier_errorbars.py"),
    ],
    "training": [
        ("federated training (synthetic)", EXP / "run_training.py", ["--rounds", "80", "--hidden", "64"]),
        ("training curves figure", PLOTS / "training_curves.py", ["training_synthetic"]),
    ],
    "utility": [
        ("utility comparison (synthetic)", EXP / "run_utility_comparison.py", ["--rounds", "80"]),
        ("utility comparison figure", PLOTS / "utility_comparison.py", ["utility_comparison_synthetic"]),
    ],
    "secagg": [
        ("secagg calibration", EXP / "run_secagg_calibration.py", []),
        ("secagg figure", PLOTS / "secagg_calibration.py", []),
    ],
}

ORDER = ["gate", "pareto", "ladder", "linkability", "multiseed", "training", "utility", "secagg"]


def run_script(label: str, path: Path, args: list[str] | None = None) -> bool:
    print(f"\n{'='*70}\n>>> {label}: {path.name}\n{'='*70}")
    cmd = [sys.executable, str(path)] + (args or [])
    result = subprocess.run(cmd, cwd=str(ROOT))
    ok = result.returncode == 0
    print(f"<<< {label}: {'OK' if ok else 'FAILED (rc=%d)' % result.returncode}")
    return ok


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "all":
        stages = ORDER
    elif stage in STAGES:
        stages = [stage]
    else:
        print(f"unknown stage '{stage}'. choices: {', '.join(ORDER)}, all")
        sys.exit(2)

    failures = []
    for s in stages:
        for entry in STAGES[s]:
            label, path = entry[0], entry[1]
            args = entry[2] if len(entry) > 2 else None
            if not run_script(label, path, args):
                failures.append(label)

    print(f"\n{'#'*70}")
    if failures:
        print(f"REPRODUCE: {len(failures)} stage(s) FAILED: {failures}")
        sys.exit(1)
    print("REPRODUCE: all stages completed. Results in artifacts/results/, "
          "figures in artifacts/figures/.")


if __name__ == "__main__":
    main()
