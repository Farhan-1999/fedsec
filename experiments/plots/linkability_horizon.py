"""Linkability horizon figure: anonymity collapses as rounds accumulate.

Plots fraction-uniquely-identifiable vs horizon for each tier count, showing the
multi-round leakage: secure aggregation hides contents, but repeated
participation erodes anonymity, faster with more tiers.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("artifacts/results/linkability.csv")
OUT = Path("artifacts/figures/linkability_horizon.png")


def main():
    with open(CSV) as f:
        rows = list(csv.DictReader(f))

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"K=3": "#3c8c40", "K=5": "#2c6fbb", "K=8": "#bb4d2c"}

    for cfg, pts in sorted(by_cfg.items()):
        pts = sorted(pts, key=lambda r: int(r["horizon"]))
        h = [int(r["horizon"]) for r in pts]
        uniq = [100 * float(r["fraction_unique"]) for r in pts]
        meanl = [float(r["mean_linkability"]) for r in pts]
        c = colors.get(cfg, "#444")
        ax1.plot(h, uniq, "-o", color=c, label=cfg, markersize=5)
        ax2.plot(h, meanl, "-o", color=c, label=cfg, markersize=5)

    ax1.set_xlabel("Horizon h (rounds observed)")
    ax1.set_ylabel("% devices uniquely identifiable")
    ax1.set_title("Anonymity collapse over rounds")
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="tiers")

    ax2.set_xlabel("Horizon h (rounds observed)")
    ax2.set_ylabel("Mean linkability  E[1/|E_i(h)|]")
    ax2.set_title("Mean linkability risk vs horizon")
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="tiers")

    fig.suptitle("Repeated participation erodes anonymity (SecAgg hides contents, not metadata)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
