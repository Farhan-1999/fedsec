"""Client-exclusion accounting figure: where does participation go?

Reads exclusion_accounting.csv (from run_exclusion_accounting.py) and draws, for
each scheme, a horizontal stacked bar splitting all available (rostered) clients
into: contributed, excluded-by-untrained-tier, excluded-by-m_min-suppression, and
excluded-by-secure-aggregation-dropout. Percentages are of the available total.

The story it tells: TiFL discards most clients by not training their tier each
round (speed bias); our framework discards only the tiers that fail the m_min
anonymity floor, retaining substantially more participation.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("artifacts/results")
FIGURES = Path("artifacts/figures")

# segment order (left -> right) and colors
SEGMENTS = [
    ("contributed", "Contributed", "#2E8B57"),
    ("selected_out", "Excluded: tier not trained", "#B0752A"),
    ("suppressed", "Excluded: m_min suppression", "#C6432E"),
    ("dropout", "Excluded: SecAgg dropout", "#7A7A7A"),
]
LABELS = {"ours": "Ours\n(tiered + m_min)", "tifl": "TiFL\n(speed-biased)"}
ORDER = ["ours", "tifl"]


def main():
    csv_path = RESULTS / "exclusion_accounting.csv"
    if not csv_path.exists():
        print(f"no CSV at {csv_path}; run run_exclusion_accounting.py first")
        sys.exit(1)

    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data[row["scheme"]] = row

    schemes = [s for s in ORDER if s in data]
    fig, ax = plt.subplots(figsize=(9.5, 3.2 + 0.4 * len(schemes)))

    ypos = list(range(len(schemes)))[::-1]  # top scheme first
    for y, scheme in zip(ypos, schemes):
        row = data[scheme]
        avail = float(row["available"])
        left = 0.0
        for key, _lab, color in SEGMENTS:
            val = float(row[key])
            pct = 100.0 * val / avail
            ax.barh(y, pct, left=left, color=color, edgecolor="white", height=0.55)
            if pct >= 5:  # label only segments wide enough to read
                ax.text(left + pct / 2, y, f"{pct:.0f}%", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
            left += pct

    ax.set_yticks(ypos)
    ax.set_yticklabels([LABELS.get(s, s) for s in schemes], fontsize=11)
    ax.set_xlabel("Share of available (rostered) clients, per round summed over run")
    ax.set_xlim(0, 100)
    mmin = data.get("ours", {}).get("m_min", "?")
    ax.set_title(f"Where participation goes  (N per config identical, m_min = {mmin})")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEGMENTS]
    ax.legend(handles, [lab for _, lab, _ in SEGMENTS],
              loc="lower center", bbox_to_anchor=(0.5, -0.42),
              ncol=2, fontsize=9, frameon=False)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "exclusion_accounting.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()