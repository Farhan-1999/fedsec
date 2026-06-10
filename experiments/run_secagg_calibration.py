"""SecAgg calibration: predicted vs empirical release, and crypto-cost comparison.

Two deliverables:
  (1) PREDICTED-vs-EMPIRICAL release calibration (required validation figure):
      for a grid of (roster size, m_min), compare the analytic tier-success
      probability against the simulator's measured release rate. Agreement
      validates the dropout model.
  (2) Crypto-cost comparison: per-client bytes and latency for complete-graph vs
      sparse-graph SecAgg across tier sizes -- the table showing why sparse
      topology matters at scale.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from dtfl.protocol.dropout import DropoutRates, apply_dropout
from dtfl.protocol.release import decide_release
from dtfl.protocol.threshold import reconstruction_threshold
from dtfl.rng import RngHub
from dtfl.secagg import (
    complete_graph_cost,
    sparse_graph_cost,
    tier_success_probability,
)

OUT_CALIB = Path("artifacts/results/secagg_calibration.csv")
OUT_COST = Path("artifacts/results/secagg_cost.csv")


def empirical_release_rate(roster_size, m_min, rates, trials=2000, seed=0):
    """Monte-Carlo release rate: draw dropout, apply the real release gate."""
    hub = RngHub(seed=seed)
    successes = 0
    for i in range(trials):
        oc = apply_dropout(roster_size, rates, hub.stream(f"t{i}"))
        t = reconstruction_threshold(oc.active_count, m_min, rates)
        # threshold uses active count; release gate checks active>=m_min and unmask>=t
        t_roster = reconstruction_threshold(roster_size, m_min, rates)
        dec = decide_release(oc, m_min, t_roster)
        successes += int(dec.released)
    return successes / trials


def calibration(rates: DropoutRates):
    print("=" * 72)
    print("PREDICTED vs EMPIRICAL RELEASE CALIBRATION")
    print("=" * 72)
    print(f"{'n':>5} {'m_min':>6} {'t':>4} {'predicted':>10} {'empirical':>10} {'|diff|':>7}")
    print("-" * 72)
    rows = []
    max_diff = 0.0
    for n in [20, 40, 60, 80, 120]:
        for m_min in [16, 32, 48]:
            if m_min > n:
                continue
            t = reconstruction_threshold(n, m_min, rates)
            pred = tier_success_probability(n, m_min, t, rates).prob_success
            emp = empirical_release_rate(n, m_min, rates, trials=3000, seed=n * 100 + m_min)
            diff = abs(pred - emp)
            max_diff = max(max_diff, diff)
            print(f"{n:>5} {m_min:>6} {t:>4} {pred:>10.3f} {emp:>10.3f} {diff:>7.3f}")
            rows.append({"n": n, "m_min": m_min, "t": t,
                         "predicted": pred, "empirical": emp, "abs_diff": diff})
    print("-" * 72)
    print(f"max |predicted - empirical| = {max_diff:.4f}")
    OUT_CALIB.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CALIB, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote -> {OUT_CALIB}")
    return max_diff


def cost_comparison(model_dim=50000):
    print("\n" + "=" * 72)
    print(f"CRYPTO COST: complete vs sparse  (model_dim={model_dim})")
    print("=" * 72)
    print(f"{'n':>6} {'topo':>9} {'degree':>7} {'KB/client':>10} {'setup_ms':>9} {'upload_ms':>10}")
    print("-" * 72)
    rows = []
    for n in [10, 50, 100, 500, 1000, 5000]:
        for cost in (complete_graph_cost(n, model_dim), sparse_graph_cost(n, model_dim)):
            print(f"{n:>6} {cost.topology:>9} {cost.degree:>7} "
                  f"{cost.per_client_bytes/1e3:>10.1f} "
                  f"{cost.setup_latency_sec*1e3:>9.1f} {cost.upload_latency_sec*1e3:>10.1f}")
            rows.append({"n": n, "topology": cost.topology, "degree": cost.degree,
                         "kb_per_client": cost.per_client_bytes/1e3,
                         "setup_ms": cost.setup_latency_sec*1e3,
                         "upload_ms": cost.upload_latency_sec*1e3})
    OUT_COST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_COST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("-" * 72)
    print(f"wrote -> {OUT_COST}")


def main():
    rates = DropoutRates(rho_mask=0.10, rho_unmask=0.05)
    calibration(rates)
    cost_comparison()


if __name__ == "__main__":
    main()
