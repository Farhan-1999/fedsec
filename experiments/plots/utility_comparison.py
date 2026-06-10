"""Utility comparison figure: does privacy machinery hurt training?

Overlays the three controlled configurations (fedavg, tiered_only, full_privacy)
on shared axes so the accuracy and time-to-accuracy gaps are read directly. All
three came from the identical engine/seed/data, so any gap is purely the cost of
the privacy mechanism.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("artifacts/results")
FIGURES = Path("artifacts/figures")

COLORS = {"fedavg": "#888780", "tiered_only": "#1D9E75", "tifl": "#BA7517", "tifl_adaptive": "#D4537E", "full_privacy": "#378ADD"}
LABELS = {"fedavg": "FedAvg (1 tier, no suppression)",
          "tiered_only": "Tiered only (no suppression)",
          "tifl": "TiFL (speed-biased, no privacy)",
          "tifl_adaptive": "TiFL adaptive (credit-based)",
          "full_privacy": "Full privacy (tiered + m_min)"}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "utility_comparison_synthetic"
    csv_path = RESULTS / f"{name}.csv"
    if not csv_path.exists():
        print(f"no CSV at {csv_path}; run run_utility_comparison.py first")
        sys.exit(1)

    runs = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            r = runs[row["config"]]
            r["round"].append(int(row["round"]))
            r["acc"].append(float(row["val_accuracy"]))
            r["loss"].append(float(row["val_loss"]))
            r["vtime"].append(float(row["virtual_time"]))
            r["rtime"].append(float(row.get("round_time", row["virtual_time"])))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    order = ["fedavg", "tiered_only", "tifl", "tifl_adaptive", "full_privacy"]

    for cfg in order:
        if cfg not in runs:
            continue
        r = runs[cfg]
        c = COLORS[cfg]
        lab = LABELS[cfg]
        axes[0].plot(r["round"], r["acc"], "-", color=c, label=lab, linewidth=1.8)
        axes[1].plot(r["round"], r["loss"], "-", color=c, label=lab, linewidth=1.8)
        axes[2].plot(r["vtime"], r["acc"], "-o", color=c, label=lab, markersize=3)
        axes[3].plot(r["rtime"], r["acc"], "-o", color=c, label=lab, markersize=3)

    axes[0].set(xlabel="Round", ylabel="Validation accuracy",
                title="Accuracy vs round")
    axes[1].set(xlabel="Round", ylabel="Validation loss", title="Loss vs round")
    axes[2].set(xlabel="Simulated time (deadline units)", ylabel="Validation accuracy",
                title="Time-to-accuracy (fixed budget)")
    axes[3].set(xlabel="Total straggler-aware time", ylabel="Validation accuracy",
                title="Total time-to-accuracy (straggler model)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Privacy machinery vs vanilla FedAvg — identical engine, seed, data, model init",
                 fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}_curves.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
