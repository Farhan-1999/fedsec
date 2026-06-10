"""SecAgg calibration figures: predicted-vs-empirical release, and cost vs n."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CALIB = Path("artifacts/results/secagg_calibration.csv")
COST = Path("artifacts/results/secagg_cost.csv")
OUT = Path("artifacts/figures/secagg_calibration.png")


def main():
    with open(CALIB) as f:
        cal = list(csv.DictReader(f))
    with open(COST) as f:
        cost = list(csv.DictReader(f))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: predicted vs empirical (should lie on y=x)
    pred = [float(r["predicted"]) for r in cal]
    emp = [float(r["empirical"]) for r in cal]
    ax1.plot([0, 1], [0, 1], "--", color="#999", label="ideal (y=x)")
    ax1.scatter(pred, emp, color="#2c6fbb", s=40, zorder=3)
    ax1.set_xlabel("Predicted release probability")
    ax1.set_ylabel("Empirical release rate")
    ax1.set_title("Dropout model validation")
    ax1.set_xlim(0.9, 1.005); ax1.set_ylim(0.9, 1.005)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # right: setup latency vs tier size, complete vs sparse (log y)
    by_topo = defaultdict(list)
    for r in cost:
        by_topo[r["topology"]].append((int(r["n"]), float(r["setup_ms"])))
    colors = {"complete": "#bb4d2c", "sparse": "#2c6fbb"}
    for topo, pts in by_topo.items():
        pts.sort()
        x = [p[0] for p in pts]; y = [p[1] for p in pts]
        ax2.plot(x, y, "-o", color=colors.get(topo, "#444"), label=topo, markersize=5)
    ax2.set_xlabel("Tier size n")
    ax2.set_ylabel("Per-client setup latency (ms)")
    ax2.set_title("SecAgg cost: complete vs sparse")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.legend(); ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("Secure-aggregation cost model: validated dropout prediction + topology scaling")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
