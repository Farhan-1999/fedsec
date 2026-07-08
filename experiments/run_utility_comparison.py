"""Controlled utility comparison: does the privacy machinery hurt FL training?

The central utility claim of the paper: adding deadline tiering + m_min
suppression + secure aggregation does NOT meaningfully degrade time-to-accuracy
or final accuracy versus vanilla FedAvg.

The clean way to prove this -- cleaner than citing external numbers from other
hardware -- is that vanilla FedAvg is a SPECIAL CASE of our own engine: one tier
covering everyone (no response-time stratification) with m_min=1 (no
suppression) and the FedAvg-equivalent merge reduces exactly to FedAvg. So we run
every configuration through the IDENTICAL engine, model, data shards, and seeds,
and the only thing that changes is the privacy machinery. Any accuracy/time gap
is therefore attributable solely to the privacy mechanism, not to confounds.

Configurations compared (all share seed, population, data, model init):
  - fedavg        : 1 tier, m_min=1            (vanilla synchronous FedAvg baseline)
  - tiered_only   : K tiers, m_min=1           (tiering, but no privacy suppression)
  - full_privacy  : K tiers, m_min=M           (tiering + suppression = our framework)

We report per-round accuracy/loss and both time axes (simulated deadline time and
measured wall-clock), so the plot shows time-to-accuracy for all three on shared
axes.
"""
from __future__ import annotations

import config as CFG

import argparse
import csv
import sys
from pathlib import Path

from dtfl.latent import LatentConfig
_STRAGGLER = {"on": False}
from dtfl.learning import (
    FedTrainConfig,
    NumpySoftmaxModel,
    federated_train,
    make_synthetic_classification,
)
from dtfl.controller.tifl import AdaptiveTiFLSelector, TiFLSelector
from dtfl.rng import RngHub
from dtfl.sim import Engine, EngineConfig, RoundConfig

RESULTS = Path("artifacts/results")


def build_data(which, hub):
    if which == "synthetic":
        train, val = make_synthetic_classification(8000, 30, 8, hub.stream("data"), separation=1.4)
        return train, val, 30, 8, "mlp"
    if which in ("cifar10", "cifar100"):
        from dtfl.learning import load_real_dataset
        train, val = load_real_dataset(which)
        C = 10 if which == "cifar10" else 100
        return train, val, train.X.shape[1], C, "cnn"
    raise ValueError(which)


def make_model(arch, d_in, C, hidden, use_torch, device='auto'):
    if use_torch:
        from dtfl.learning import get_torch_model
        img = (3, 32, 32) if arch == "cnn" else None
        return get_torch_model(input_dim=d_in, num_classes=C, arch=arch,
                               hidden_dim=hidden, image_shape=img, seed=0, device=device)
    return NumpySoftmaxModel(d_in, C, hidden_dim=hidden, seed=0)


def run_config(name, num_tiers, m_min, *, train, val, d_in, C, arch, use_torch,
               hidden, lr, rounds, devices, seed, tier_selector=None, device='auto'):
    """Run one configuration through the shared engine. Same seed => same
    population, availability, latency draws, and data shards across configs."""
    lcfg = (LatentConfig(class_separation=0.7, tail_prob=0.12, tail_scale_log=0.9)
            if _STRAGGLER["on"] else LatentConfig())
    eng = Engine(
        EngineConfig(seed=seed, num_devices=devices, num_rounds=rounds,
                     round_config=RoundConfig(m_min=m_min),
                     flhetbench=CFG.population_config()),
        lcfg,
    )
    # Deadlines: adaptive quantile-tracking controller (the deployable server
    # adapts deadlines online). K=1 is a single covering deadline -> no tiering.
    if num_tiers == 1:
        cutoffs = eng.calibrate_fixed_deadlines(())  # single covering cutoff
        controller = None
    else:
        qs = tuple((k + 1) / num_tiers for k in range(num_tiers - 1))
        controller = eng.adaptive_policy(qs)
        cutoffs = tuple(controller._cutoffs)  # round-0 init + time-budget scale

    model = make_model(arch, d_in, C, hidden, use_torch, device=device)
    res = federated_train(
        model, train, val, eng, cutoffs,
        FedTrainConfig(local_epochs=1, local_lr=lr, server_lr=1.0),
        RoundConfig(m_min=m_min), base_seed=7, tier_selector=tier_selector,
        controller=controller,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=CFG.DATASET, choices=["synthetic", "cifar10", "cifar100"])
    ap.add_argument("--rounds", type=int, default=CFG.TRAINING_ROUNDS)
    ap.add_argument("--hidden", type=int, default=CFG.HIDDEN)
    ap.add_argument("--lr", type=float, default=CFG.LR)
    ap.add_argument("--devices", type=int, default=CFG.DEVICES)
    ap.add_argument("--tiers", type=int, default=CFG.TIERS)
    ap.add_argument("--m_min", type=int, default=8)
    ap.add_argument("--seed", type=int, default=CFG.SEED)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--stragglers", action="store_true", help="heavy slow-device tail")
    args = ap.parse_args()

    _STRAGGLER["on"] = args.stragglers
    from dtfl.learning import resolve_device
    use_torch = args.data != "synthetic"
    if use_torch:
        print(f"[device] requested={args.device} resolved={resolve_device(args.device)}")
    hub = RngHub(seed=args.seed)
    train, val, d_in, C, arch = build_data(args.data, hub)
    print(f"data={args.data} train={len(train)} val={len(val)} d_in={d_in} C={C} arch={arch}")
    print(f"comparing on IDENTICAL seed={args.seed}, population, data, model init\n")

    tifl_selector = TiFLSelector(args.tiers, speed_bias=1.0, floor=0.10)
    tifl_adaptive = AdaptiveTiFLSelector(args.tiers, interval=5, seed=args.seed)
    configs = [
        ("fedavg", 1, 1, None),
        ("tiered_only", args.tiers, 1, None),
        ("tifl", args.tiers, 1, tifl_selector),
        ("tifl_adaptive", args.tiers, 1, tifl_adaptive),
        ("full_privacy", args.tiers, args.m_min, None),
    ]

    runs = []
    for name, K, m, sel in configs:
        tag = "single-tier/round" if sel is not None else "all tiers merged"
        print(f"=== {name}  (tiers={K}, m_min={m}, {tag}) ===")
        res = run_config(
            name, K, m,
            train=train, val=val, d_in=d_in, C=C, arch=arch, use_torch=use_torch,
            hidden=args.hidden, lr=args.lr, rounds=args.rounds,
            devices=args.devices, seed=args.seed, tier_selector=sel, device=args.device,
        )
        runs.append((name, res))
        acc = res.val_accuracy
        print(f"  final acc={acc[-1]:.3f}  max acc={max(acc):.3f}  "
              f"final loss={res.val_loss[-1]:.3f}  wall={res.wall_time[-1]:.1f}s")
        for tgt in (0.3, 0.4, 0.5):
            r = res.round_to_accuracy(tgt)
            st = res.straggler_time_to_accuracy(tgt)
            print(f"  acc>={tgt}: round={r}  total_time={st if st is None else round(st,1)}")
        print()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"utility_comparison_{args.data}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "round", "val_accuracy", "val_loss",
                    "virtual_time", "wall_time", "round_time", "participation"])
        for name, res in runs:
            for i in range(len(res.val_accuracy)):
                w.writerow([name, i, res.val_accuracy[i], res.val_loss[i],
                            res.virtual_time[i], res.wall_time[i], res.round_time[i], res.participation[i]])
    print(f"wrote -> {out}")

    # quick headline gap summary
    print("\n--- utility gap vs fedavg (final accuracy) ---")
    base = dict(runs)["fedavg"].val_accuracy[-1]
    for name, res in runs:
        gap = res.val_accuracy[-1] - base
        print(f"  {name:>14}: {res.val_accuracy[-1]:.3f}  (gap {gap:+.3f})")


if __name__ == "__main__":
    main()