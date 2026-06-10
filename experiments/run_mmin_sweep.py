"""m_min sweep: privacy/timing trade-off vs FedAvg and TiFL reference points.

Answers the contribution question directly: as we vary the privacy knob m_min,
how does our framework's TOTAL straggler-aware time-to-accuracy move between the
FedAvg point (no tiering, slow, leaky-free but exposes nothing because untiered)
and the TiFL point (aggressive tiering, fast, exposes capability)?

Produces two trade-off views, all reaching the same target accuracy:
  (1) total straggler-aware time to reach acc>=X, as a function of m_min, with
      FedAvg and TiFL drawn as horizontal reference lines.
  (2) number of ROUNDS to reach acc>=X for every framework (bar-style series).

All configurations run through the IDENTICAL engine / seed / data / model init
(see run_utility_comparison), so differences are attributable only to the method.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dtfl.controller.tifl import AdaptiveTiFLSelector, TiFLSelector
from dtfl.learning import make_synthetic_classification
from dtfl.rng import RngHub

import config as CFG
import run_utility_comparison as U

RESULTS = Path("artifacts/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=CFG.TRAINING_ROUNDS)
    ap.add_argument("--tiers", type=int, default=CFG.TIERS)
    ap.add_argument("--target", type=float, default=CFG.TARGET_ACC)
    ap.add_argument("--devices", type=int, default=CFG.DEVICES)
    ap.add_argument("--seed", type=int, default=CFG.SEED)
    ap.add_argument("--hidden", type=int, default=CFG.HIDDEN)
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "cifar10", "cifar100"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--lr", type=float, default=CFG.LR)
    ap.add_argument("--mmins", type=str, default="1,4,8,12,16,20,24,28,32,36,40,44,48,52",
                    help="comma-separated m_min values to sweep")
    ap.add_argument("--time_cap", type=float, default=700.0,
                    help="stop sweeping once total time-to-acc exceeds this")
    args = ap.parse_args()

    # straggler scenario on (the whole point is heterogeneous device speed)
    U._STRAGGLER["on"] = True
    hub = RngHub(seed=args.seed)
    use_torch = args.data != "synthetic"
    train, val, d_in, C, arch = U.build_data(args.data, hub)
    print(f"straggler scenario | train={len(train)} val={len(val)} target=acc>={args.target}\n")

    common = dict(train=train, val=val, d_in=d_in, C=C, arch=arch, use_torch=use_torch, device=args.device,
                  hidden=args.hidden, lr=args.lr, rounds=args.rounds,
                  devices=args.devices, seed=args.seed)

    def metrics(res):
        return (res.straggler_time_to_accuracy(args.target),
                res.round_to_accuracy(args.target),
                res.val_accuracy[-1])

    rows = []

    # --- reference points ---
    print("=== reference: FedAvg (1 tier, no tiering) ===")
    t, rd, fa = metrics(U.run_config("fedavg", 1, 1, **common))
    print(f"  total_time={t}  rounds={rd}  final_acc={fa:.3f}")
    rows.append({"method": "fedavg", "m_min": "", "total_time": t, "rounds": rd, "final_acc": fa})

    print("=== reference: TiFL (speed-biased) ===")
    sel = TiFLSelector(args.tiers, speed_bias=1.0, floor=0.10)
    t, rd, fa = metrics(U.run_config("tifl", args.tiers, 1, tier_selector=sel, **common))
    print(f"  total_time={t}  rounds={rd}  final_acc={fa:.3f}")
    rows.append({"method": "tifl", "m_min": "", "total_time": t, "rounds": rd, "final_acc": fa})

    print("=== reference: TiFL adaptive ===")
    sel = AdaptiveTiFLSelector(args.tiers, interval=5, seed=args.seed)
    t, rd, fa = metrics(U.run_config("tifl_adaptive", args.tiers, 1, tier_selector=sel, **common))
    print(f"  total_time={t}  rounds={rd}  final_acc={fa:.3f}")
    rows.append({"method": "tifl_adaptive", "m_min": "", "total_time": t, "rounds": rd, "final_acc": fa})

    # --- our framework swept over m_min ---
    print("\n=== our framework: m_min sweep ===")
    mmins = [int(x) for x in args.mmins.split(",")]
    for m in mmins:
        t, rd, fa = metrics(U.run_config(f"ours_m{m}", args.tiers, m, **common))
        print(f"  m_min={m:>3}: total_time={t}  rounds={rd}  final_acc={fa:.3f}")
        rows.append({"method": "ours", "m_min": m, "total_time": t, "rounds": rd, "final_acc": fa})
        # stop once we exceed the time cap (or the target stops being reachable)
        if t is None or t > args.time_cap:
            print(f"  (stopping sweep: m_min={m} exceeded time cap {args.time_cap}s "
                  f"or target unreachable)")
            break

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "mmin_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "m_min", "total_time", "rounds", "final_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
