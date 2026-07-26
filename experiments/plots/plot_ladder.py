"""Adversary-ladder figure (three rungs: L0, L1, L2).

The stored CSVs use the legacy 4-column schema (L0,L1,L2,L3) where the old L2 was
a Bayes-optimal rung that has been removed from the paper. This script DROPS the
old L2 column and RENAMES the old L3 (omniscient) to L2, so the figure shows the
current three-rung ladder: L0 (unsupervised) < L1 (lightly supervised) < L2
(omniscient ceiling).
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("artifacts/results")
FIGURES = Path("artifacts/figures")

# current rungs -> source column in the legacy CSV
SRC = {"L0": "L0", "L1": "L1", "L2": "L3"}   # old L3 (omniscient) becomes L2
RUNGS = ["L0", "L1", "L2"]
RUNG_LABELS = {"L0": "L0\nunsupervised", "L1": "L1\nlight superv.",
               "L2": "L2\nomniscient"}
RUNG_COLORS = {"L0": "#8C8C8C", "L1": "#2E7D9A", "L2": "#5B3A8C"}


def load_multiseed(path):
    row = next(r for r in csv.DictReader(open(path)) if r["point"] == "undefended")
    means = [float(row[f"{SRC[g]}_mean"]) for g in RUNGS]
    cis = [float(row[f"{SRC[g]}_ci_half"]) for g in RUNGS]
    return means, cis


def load_single(path):
    rows = list(csv.DictReader(open(path)))
    defenses = [r["defense"] for r in rows]
    mmins = [int(r["m_min"]) for r in rows]
    series = {g: [float(r[SRC[g]]) for r in rows] for g in RUNGS}
    return defenses, mmins, series


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    means, cis = load_multiseed(RESULTS / "multiseed_ladder.csv")
    x = np.arange(len(RUNGS))
    axL.bar(x, means, yerr=cis, capsize=6, color=[RUNG_COLORS[g] for g in RUNGS],
            edgecolor="black", linewidth=0.6, width=0.6,
            error_kw=dict(elinewidth=1.5, ecolor="#333"))
    for xi, m, c in zip(x, means, cis):
        axL.text(xi, m + c + 0.02, f"{m:.2f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels([RUNG_LABELS[g] for g in RUNGS], fontsize=10)
    axL.set_ylabel("Capability-inference advantage"); axL.set_ylim(0, 1.12)
    axL.set_title("Adversary ladder, undefended\n(mean \u00b1 95% CI, 5 seeds)")
    axL.grid(True, axis="y", alpha=0.3)
    axL.annotate("L1 recovers most of\nthe omniscient advantage:\ncheaply extractable",
                 xy=(1, means[1]), xytext=(0.75, means[1] + 0.20),
                 ha="center", fontsize=9.5, color="#1B4A5A", fontstyle="italic",
                 arrowprops=dict(arrowstyle="->", color="#1B4A5A", lw=1.2))

    defenses, mmins, series = load_single(RESULTS / "ladder.csv")
    xd = np.arange(len(defenses))
    for g in RUNGS:
        axR.plot(xd, series[g], "-o", color=RUNG_COLORS[g], markersize=7,
                 linewidth=2, label=g)
    axR.set_xticks(xd)
    axR.set_xticklabels([f"{d}\n(m_min={m})" for d, m in zip(defenses, mmins)], fontsize=10)
    axR.set_ylabel("Capability-inference advantage"); axR.set_ylim(0, 1.12)
    axR.set_title("Every rung falls as the\nanonymity floor rises")
    axR.grid(True, alpha=0.3); axR.legend(title="adversary", fontsize=10, loc="upper right")

    fig.suptitle("Capability leakage is real and cheaply extractable \u2014 and the "
                 "anonymity floor closes it", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "adversary_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()