"""Training curves: loss, accuracy, convergence, and time-to-accuracy.

Reads a training CSV (from run_training.py) and renders a 2x2 panel:
  - accuracy vs round         (learning / convergence curve)
  - loss vs round             (loss curve)
  - accuracy vs wall-clock    (measured training-time curve)
  - accuracy vs virtual time  (time-to-accuracy, the systems metric)

Overlays multiple runs if the CSV has more than one 'run' (e.g. --compare).
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


def load(csv_path: Path):
    runs = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            r = runs[row["run"]]
            r["round"].append(int(row["round"]))
            r["acc"].append(float(row["val_accuracy"]))
            r["loss"].append(float(row["val_loss"]))
            r["vtime"].append(float(row["virtual_time"]))
            r["wtime"].append(float(row["wall_time"]))
    return runs


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "training_synthetic"
    csv_path = RESULTS / f"{name}.csv"
    if not csv_path.exists():
        print(f"no CSV at {csv_path}; run experiments/run_training.py first")
        sys.exit(1)
    runs = load(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ["#2c6fbb", "#bb4d2c", "#3c8c40", "#7a4dbb"]

    for i, (label, r) in enumerate(runs.items()):
        c = colors[i % len(colors)]
        axes[0, 0].plot(r["round"], r["acc"], "-", color=c, label=label)
        axes[0, 1].plot(r["round"], r["loss"], "-", color=c, label=label)
        axes[1, 0].plot(r["wtime"], r["acc"], "-o", color=c, label=label, markersize=3)
        axes[1, 1].plot(r["vtime"], r["acc"], "-o", color=c, label=label, markersize=3)

    axes[0, 0].set(xlabel="Round", ylabel="Validation accuracy",
                   title="Accuracy / convergence curve")
    axes[0, 1].set(xlabel="Round", ylabel="Validation loss", title="Loss curve")
    axes[1, 0].set(xlabel="Measured wall-clock (s)", ylabel="Validation accuracy",
                   title="Training-time curve (measured)")
    axes[1, 1].set(xlabel="Simulated time (deadline units)", ylabel="Validation accuracy",
                   title="Time-to-accuracy (systems metric)")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        if len(runs) > 1:
            ax.legend()

    fig.suptitle(f"Federated training curves — {name}", fontsize=13)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}_curves.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
