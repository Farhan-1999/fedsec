"""Multi-seed frontier figure with 95% CI error bars on both axes.

Shows the privacy-utility frontier as mean points with horizontal (participation)
and vertical (advantage) confidence intervals. The widening bars through the
cliff region are the visual statement that the trade-off is well-defined in
expectation but high-variance near practical operating points.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("artifacts/results/multiseed_frontier.csv")
OUT = Path("artifacts/figures/frontier_errorbars.png")


def main():
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["part_mean"]))

    x = [float(r["part_mean"]) for r in rows]
    y = [float(r["adv_mean"]) for r in rows]
    xerr = [float(r["part_ci_half"]) for r in rows]
    yerr = [float(r["adv_ci_half"]) for r in rows]
    labels = [r["point"] for r in rows]
    mmins = [r["m_min"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt="o-", color="#2c6fbb", ecolor="#b0532c",
        elinewidth=1.5, capsize=4, markersize=6, linewidth=1.5,
        label="mean ± 95% CI (5 seeds)",
    )
    for xi, yi, lab, m in zip(x, y, labels, mmins):
        ax.annotate(f"{lab}\n(m={m})", (xi, yi), fontsize=7,
                    xytext=(6, 6), textcoords="offset points")

    ax.set_xlabel("Mean participation per round  (utility →)")
    ax.set_ylabel("L1 capability-inference advantage  (← privacy)")
    ax.set_title("Privacy–utility frontier with 95% CIs\n(widening bars = high variance through the cliff)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.05, 0.75)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
