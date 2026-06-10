"""Headline Pareto figure: attacker advantage vs utility, per defense point.

Reads the sweep CSV and plots advantage (privacy: lower = better) against a
utility proxy (mean participation, or release rate). Each marker is a defense
configuration; the frontier is the lower-left-to-upper-right envelope showing
the exchange rate between hidden capability and participation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("artifacts/results/headline_pareto.csv")
OUT = Path("artifacts/figures/headline_pareto.png")


def load_rows():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    # Use the single-bucket points (bucket axis shown inert; avoids double-plotting).
    # Group by K so each tier-count traces its own frontier.
    by_k: dict[int, list] = {}
    for r in rows:
        if not r["bucket"].startswith("single"):
            continue
        k = int(r["num_tiers"])
        by_k.setdefault(k, []).append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {5: "#2c6fbb", 8: "#bb4d2c", 3: "#3c8c40", 12: "#7a4dbb", 16: "#999999"}
    for k, pts in sorted(by_k.items()):
        pts = sorted(pts, key=lambda r: float(r["mean_participation"]))
        x = [float(r["mean_participation"]) for r in pts]
        y = [float(r["advantage"]) for r in pts]
        c = colors.get(k, "#444444")
        ax.plot(x, y, "-o", color=c, label=f"K={k}", markersize=5, linewidth=1.5)
        # annotate m_min at each point
        for r in pts:
            ax.annotate(
                r["m_min"],
                (float(r["mean_participation"]), float(r["advantage"])),
                fontsize=6, color=c, xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlabel("Mean participation per round  (utility →)")
    ax.set_ylabel("L1 capability-inference advantage  (← privacy)")
    ax.set_title("Privacy–utility frontier (labels = m_min)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="tiers")
    ax.set_ylim(-0.05, 0.8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
