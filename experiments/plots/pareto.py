"""Headline Pareto figure: adversary advantage vs utility, per defense point.

Reads the sweep CSV and plots capability-inference advantage (privacy: lower =
better) against a utility proxy (mean participation). Each marker is a defense
configuration; together they trace the exchange rate between hidden capability
and participation.

THREE ADVERSARY LEVELS are drawn on shared axes, all evaluated on the SAME
simulated transcripts so they differ only in what the adversary knows:

  L1  server + colluding clients (PRIMARY, the realistic deployment threat).
      An honest-but-curious server colluding with a fraction x of clients. The
      colluders self-assign to tiers, so the adversary knows their tier exactly
      and can subtract them from each published count to expose the honest
      anonymity set, then intersect signatures across rounds. Requires no
      physical access and breaks no cryptography.
  L2  released-round observer. Attributes a target's tier on any round that
      released. Stronger per-device visibility than L1, but no colluders.
  L3  omniscient ceiling. Given the capability labels of every other device --
      an upper bound on what this channel can yield to any adversary.

Reporting all three converts "an attack works" into "we characterized leakage
across adversary strength", and the L1-to-L3 gap shows how much of the available
leakage the realistic adversary actually captures.

SCOPE: the figure reports the K=5 operating configuration only -- the tier count
used throughout the evaluation, whose per-tier mass (~77 at the measured mean
availability) sets the anonymity-floor operating range. The K=8 sweep is not
plotted.

BACKWARD COMPATIBILITY: CSVs written before the multi-level harness contain only
the ``advantage`` column. Levels whose columns are absent are simply skipped, so
this script still works on older sweeps (drawing L1 alone).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("artifacts/results/headline_pareto.csv")
OUT = Path("artifacts/figures/headline_pareto.png")

# Tier count plotted. The evaluation's operating configuration.
PLOT_K = 5

# level -> (csv column, label, color, marker)
LEVELS = [
    ("L1", "advantage", "L1: server + colluding clients", "#2c6fbb", "o"),
    ("L2", "advantage_l2", "L2: released-round observer", "#BA7517", "s"),
    ("L3", "advantage_l3", "L3: omniscient ceiling", "#5B3A8C", "^"),
]


def load_rows():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    # Use the single-bucket points (bucket axis shown inert; avoids double-plotting).
    pts = [
        r
        for r in rows
        if r["bucket"].startswith("single") and int(r["num_tiers"]) == PLOT_K
    ]
    if not pts:
        raise SystemExit(f"no K={PLOT_K} single-bucket points found in {CSV}")

    pts = sorted(pts, key=lambda r: float(r["mean_participation"]))
    x = [float(r["mean_participation"]) for r in pts]

    fig, ax = plt.subplots(figsize=(7.4, 5))
    drawn = []
    for level, col, label, color, marker in LEVELS:
        if col not in pts[0]:
            continue  # older CSV without this level
        y = [float(r[col]) for r in pts]
        ax.plot(x, y, "-", color=color, marker=marker, markersize=5,
                linewidth=1.5, label=label)
        drawn.append(level)

    # annotate m_min once, on the primary (L1) series, to avoid clutter
    for r in pts:
        ax.annotate(
            r["m_min"],
            (float(r["mean_participation"]), float(r["advantage"])),
            fontsize=6, color=LEVELS[0][3], xytext=(3, 3), textcoords="offset points",
        )

    # collusion rate, if the harness recorded it, for the caption
    xrate = None
    if "collusion_rate" in pts[0]:
        try:
            xrate = float(pts[0]["collusion_rate"])
        except (TypeError, ValueError):
            xrate = None

    ax.set_xlabel("Mean participation per round  (utility →)")
    ax.set_ylabel("Capability-inference advantage  (← privacy)")
    sub = f", collusion rate x={xrate:.2f}" if xrate is not None else ""
    ax.set_title(
        f"Privacy–utility frontier across adversary levels\n"
        f"(K={PLOT_K}, point labels = m_min{sub})",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="adversary level", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"plotted K={PLOT_K}: {len(pts)} points, levels {'+'.join(drawn)}, "
          f"m_min {min(int(r['m_min']) for r in pts)}..{max(int(r['m_min']) for r in pts)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()