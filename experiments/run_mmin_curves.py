"""m_min time-to-accuracy curves for OUR framework only.

Unlike run_mmin_sweep (which reduces each m_min to a single scalar time-to-target),
this experiment records the full accuracy-vs-time TRAJECTORY of our framework at
several m_min values and overlays them, so the effect of the anonymity floor on
time-to-accuracy is read directly as a family of curves.

No FedAvg / TiFL baselines here: the comparison is our framework against itself
across the privacy knob. All runs share the identical engine / seed / data / model
init, so any difference between curves is attributable solely to m_min.

Time axis is the TRUE straggler-aware time (round_time), not simulated deadline
units.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dtfl.rng import RngHub

import config as CFG
import run_utility_comparison as U

RESULTS = Path("artifacts/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=CFG.TRAINING_ROUNDS)
    ap.add_argument("--tiers", type=int, default=CFG.TIERS)
    ap.add_argument("--devices", type=int, default=CFG.DEVICES)
    ap.add_argument("--seed", type=int, default=CFG.SEED)
    ap.add_argument("--hidden", type=int, default=CFG.HIDDEN)
    ap.add_argument("--data", default=CFG.DATASET, choices=["synthetic", "cifar10", "cifar100"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--lr", type=float, default=CFG.LR)
    ap.add_argument("--mmins", type=str, default="1,40,55,70,77,85,100",
                    help="m_min values whose curves to overlay; default spans the "
                         "true privacy cliff at N=1500, K=5, measured availability "
                         "(~0.26 -> ~77/tier; cliff between 77 and 85)")
    args = ap.parse_args()

    # Heterogeneous straggler scenario: the regime where m_min actually matters.
    U._STRAGGLER["on"] = True
    hub = RngHub(seed=args.seed)
    use_torch = args.data != "synthetic"
    train, val, d_in, C, arch = U.build_data(args.data, hub)
    print(f"our-framework m_min curves | train={len(train)} val={len(val)} "
          f"tiers={args.tiers}\n")

    common = dict(train=train, val=val, d_in=d_in, C=C, arch=arch,
                  use_torch=use_torch, device=args.device, hidden=args.hidden,
                  lr=args.lr, rounds=args.rounds, devices=args.devices, seed=args.seed)

    mmins = [int(x) for x in args.mmins.split(",")]
    rows = []
    for m in mmins:
        print(f"=== ours  (m_min={m}) ===")
        res = U.run_config(f"ours_m{m}", args.tiers, m, **common)
        acc = res.val_accuracy
        print(f"  final acc={acc[-1]:.3f}  max acc={max(acc):.3f}  "
              f"final total-time={res.round_time[-1]:.1f}")
        for i in range(len(acc)):
            rows.append({
                "m_min": m,
                "round": i,
                "val_accuracy": acc[i],
                "round_time": res.round_time[i],       # true straggler-aware time
                "participation": res.participation[i],
            })
        print()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"mmin_curves_{args.data}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["m_min", "round", "val_accuracy",
                                          "round_time", "participation"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()