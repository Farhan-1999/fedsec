"""Full real-data pipeline runner.

Runs the COMPLETE evaluation:
  - privacy experiments (gate, pareto, ladder, linkability, multiseed): these do
    NOT train a model -- they simulate timing metadata and run the attacker, so
    they are dataset-agnostic and run once at the shared device count.
  - training / utility / m_min-sweep: these DO train, and are run on BOTH
    cifar10 and cifar100, IID-partitioned across the full device population.

Everything uses the shared defaults in config.py (devices=1500, K=5, seed=42)
unless an experiment directly studies a parameter.

WARNING: real CIFAR training at 1500 devices on CPU is a multi-hour to multi-day
job. Each training experiment is launched as its own subprocess and writes its
CSV as soon as it finishes, so an interruption only loses the in-flight run --
everything already completed is preserved on disk. Re-running skips nothing
automatically; comment out stages that already finished if you restart.

Usage:
    python experiments/run_all_real.py                 # everything
    python experiments/run_all_real.py --skip-privacy   # only the training half
    python experiments/run_all_real.py --datasets cifar10   # one dataset
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
PLOTS = EXP / "plots"
PY = sys.executable


def run(label, script, args):
    """Run one experiment as a subprocess from the repo root; stream output."""
    cmd = [PY, str(script), *args]
    print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {label}\n  {' '.join(cmd)}\n{'='*70}",
          flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {label}: {status} in {dt/60:.1f} min", flush=True)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cifar10,cifar100",
                    help="comma-separated real datasets for training experiments")
    ap.add_argument("--rounds", type=int, default=100, help="training rounds per run")
    ap.add_argument("--skip-privacy", action="store_true",
                    help="skip the simulation-only privacy experiments")
    ap.add_argument("--skip-training", action="store_true",
                    help="skip the real-data training experiments")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    results = {}

    # --- PHASE 1: privacy experiments (simulation-only, dataset-agnostic) -----
    if not args.skip_privacy:
        print("\n########## PHASE 1: privacy experiments (no training) ##########")
        results["gate"] = run("gate", EXP / "run_gate.py", [])
        results["pareto"] = run("headline pareto", EXP / "run_headline_pareto.py", [])
        run("pareto figure", PLOTS / "pareto.py", [])
        results["ladder"] = run("ladder", EXP / "run_ladder.py", [])
        results["linkability"] = run("linkability", EXP / "run_linkability.py", [])
        run("linkability figure", PLOTS / "linkability_horizon.py", [])
        results["multiseed_frontier"] = run("multiseed frontier", EXP / "run_multiseed_frontier.py", [])
        run("frontier figure", PLOTS / "frontier_errorbars.py", [])
        results["multiseed_ladder_link"] = run("multiseed ladder+linkability",
                                               EXP / "run_multiseed_ladder_linkability.py", [])
        results["secagg"] = run("secagg calibration", EXP / "run_secagg_calibration.py", [])
        run("secagg figure", PLOTS / "secagg_calibration.py", [])

    # --- PHASE 2: real-data training experiments (both datasets) --------------
    if not args.skip_training:
        print("\n########## PHASE 2: real-data training (both datasets) ##########")
        for ds in datasets:
            print(f"\n---------- dataset: {ds} ----------")
            # learning curve for the framework itself
            results[f"train_{ds}"] = run(
                f"training curve [{ds}]", EXP / "run_training.py",
                ["--data", ds, "--rounds", str(args.rounds)])
            run(f"training figure [{ds}]", PLOTS / "training_curves.py", [f"training_{ds}"])

            # controlled utility comparison (FedAvg / tiered / TiFL / adaptive / ours)
            results[f"utility_{ds}"] = run(
                f"utility comparison [{ds}]", EXP / "run_utility_comparison.py",
                ["--data", ds, "--rounds", str(args.rounds)])
            run(f"utility figure [{ds}]", PLOTS / "utility_comparison.py",
                [f"utility_comparison_{ds}"])

        # m_min sweep: trains repeatedly; run once on the primary (first) dataset
        # to keep runtime bounded -- it is the trade-off SHAPE that matters.
        primary = datasets[0]
        results[f"mmin_{primary}"] = run(
            f"m_min sweep [{primary}]", EXP / "run_mmin_sweep.py",
            ["--rounds", str(args.rounds), "--data", primary])
        run("m_min sweep figure", PLOTS / "mmin_sweep.py", [])

    # --- summary --------------------------------------------------------------
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for k, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {k}")
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} experiments succeeded.")


if __name__ == "__main__":
    main()
