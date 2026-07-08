"""Training experiment: produce loss, accuracy, convergence, and time curves.

Runs real federated training under the timing/tier dynamics and records the full
per-round trajectory (loss, accuracy, simulated time, measured wall-clock,
participation). Writes a CSV consumed by experiments/plots/training_curves.py.

Three modes:
  --data synthetic   : in-memory data, no download, NumPy MLP (default; fast)
  --data cifar10     : real CIFAR-10 (needs '.[learning]' + download), torch CNN
  --compare          : run multiple controllers on the same data and overlay

Examples:
  python experiments/run_training.py
  python experiments/run_training.py --rounds 120 --hidden 128
  python experiments/run_training.py --data cifar10 --rounds 200
  python experiments/run_training.py --compare
"""
from __future__ import annotations

import config as CFG

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dtfl.controller import FixedQuantile, QuantileTrackingController
from dtfl.latent import LatentConfig
from dtfl.learning import (
    FedTrainConfig,
    NumpySoftmaxModel,
    federated_train,
    iid_shard,
    make_synthetic_classification,
)
from dtfl.rng import RngHub
from dtfl.sim import Engine, EngineConfig, RoundConfig

RESULTS = Path("artifacts/results")


def build_data(which: str, hub: RngHub):
    """Return (train, val, input_dim, num_classes, arch_hint)."""
    if which == "synthetic":
        train, val = make_synthetic_classification(
            8000, 30, 8, hub.stream("data"), separation=1.4
        )
        return train, val, 30, 8, "mlp"
    if which in ("cifar10", "cifar100"):
        from dtfl.learning import load_real_dataset
        train, val = load_real_dataset(which)
        C = 10 if which == "cifar10" else 100
        return train, val, train.X.shape[1], C, "cnn"
    raise ValueError(f"unknown data '{which}'")


def make_model(arch, input_dim, num_classes, hidden, use_torch, image_shape=None, device='auto'):
    if use_torch:
        from dtfl.learning import get_torch_model
        return get_torch_model(
            device=device,
            input_dim=input_dim, num_classes=num_classes,
            arch=arch, hidden_dim=hidden, image_shape=image_shape, seed=0,
        )
    return NumpySoftmaxModel(input_dim, num_classes, hidden_dim=hidden, seed=0)


def run_once(label, controller_factory, train, val, input_dim, num_classes,
             *, rounds, hidden, lr, use_torch, arch, num_devices, seed, device='auto'):
    lcfg = LatentConfig()
    eng = Engine(
        EngineConfig(seed=seed, num_devices=num_devices, num_rounds=rounds,
                     round_config=RoundConfig(m_min=8),
                     flhetbench=CFG.population_config()),
        lcfg,
    )
    # K interior quantiles at k/K for k=1..K-1 (calibrate appends a covering tier).
    pilot_quantiles = tuple(k / CFG.TIERS for k in range(1, CFG.TIERS))
    img_shape = (3, 32, 32) if arch == "cnn" else None
    model = make_model(arch, input_dim, num_classes, hidden, use_torch, image_shape=img_shape, device=device)

    # Adaptive quantile-tracking controller: the server adapts tier deadlines
    # online from the aggregate counts (deployable, transcript-blind). The
    # training loop now updates deadlines each round via this controller.
    controller = eng.adaptive_policy(pilot_quantiles)
    cutoffs = tuple(controller._cutoffs)  # round-0 init + time-budget scale
    res = federated_train(
        model, train, val, eng, cutoffs,
        FedTrainConfig(local_epochs=1, local_lr=lr, server_lr=1.0),
        RoundConfig(m_min=8), base_seed=7,
        controller=controller,
    )
    return label, res


class _EmptyView:
    def __len__(self): return 0
    def rounds(self): return []
    def tier_counts(self, r): return {}


def _empty_view():
    return _EmptyView()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=CFG.DATASET, choices=["synthetic", "cifar10", "cifar100"])
    ap.add_argument("--rounds", type=int, default=CFG.TRAINING_ROUNDS)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--devices", type=int, default=CFG.DEVICES)
    ap.add_argument("--seed", type=int, default=CFG.SEED)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compare", action="store_true",
                    help="overlay multiple controllers")
    args = ap.parse_args()

    from dtfl.learning import resolve_device
    use_torch = args.data != "synthetic"
    if use_torch:
        print(f"[device] requested={args.device} resolved={resolve_device(args.device)}")
    hub = RngHub(seed=args.seed)
    train, val, d_in, C, arch = build_data(args.data, hub)
    print(f"data={args.data}  train={len(train)} val={len(val)}  d_in={d_in} C={C} arch={arch}")

    # Tier count and matching cumulative-fraction targets (k/K for k=1..K, last=1.0).
    K = CFG.TIERS
    qt_targets = tuple(k / K for k in range(1, K + 1))

    runs = []
    if args.compare:
        factories = {
            "fixed_quantile": lambda pilot: FixedQuantile(K, pilot),
            "quantile_track": lambda pilot: QuantileTrackingController(
                K, qt_targets, pilot),
        }
    else:
        factories = {"train": lambda pilot: FixedQuantile(K, pilot)}

    for label, factory in factories.items():
        print(f"\n=== running: {label} ===")
        lbl, res = run_once(
            label, factory, train, val, d_in, C,
            rounds=args.rounds, hidden=args.hidden, lr=args.lr,
            use_torch=use_torch, arch=arch, num_devices=args.devices, seed=args.seed, device=args.device,
        )
        runs.append((lbl, res))
        print(f"  final acc={res.val_accuracy[-1]:.3f}  final loss={res.val_loss[-1]:.3f}  "
              f"wall={res.wall_time[-1]:.1f}s")
        for tgt in (0.4, 0.5, 0.6):
            print(f"  round-to-{tgt}={res.round_to_accuracy(tgt)}  "
                  f"time-to-{tgt}={res.time_to_accuracy(tgt)}")

    # write CSV: one row per (run, round)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"training_{args.data}{'_compare' if args.compare else ''}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "round", "val_accuracy", "val_loss",
                    "virtual_time", "wall_time", "participation"])
        for label, res in runs:
            for r in range(len(res.val_accuracy)):
                w.writerow([label, r, res.val_accuracy[r], res.val_loss[r],
                            res.virtual_time[r], res.wall_time[r], res.participation[r]])
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()