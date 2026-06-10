"""m_min sweep figure: total-time trade-off + rounds-to-accuracy.

Left: total straggler-aware time to reach the target accuracy as a function of
m_min (our framework), with FedAvg and TiFL drawn as horizontal reference lines.
Shows the operating band where suppression buys speed before the accuracy cliff.

Right: number of rounds to reach the target accuracy for every framework
(reference points + our framework at each m_min), as a bar chart.

Points where the target was never reached (cliff) are omitted from the line and
shown as a gap.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("artifacts/results")
FIGURES = Path("artifacts/figures")

REF_COLORS = {"fedavg": "#888780", "tifl": "#BA7517", "tifl_adaptive": "#D4537E"}
REF_LABELS = {"fedavg": "FedAvg", "tifl": "TiFL", "tifl_adaptive": "TiFL adaptive"}


def fnum(x):
    return float(x) if x not in ("", "None", None) else None


def main():
    csv_path = RESULTS / "mmin_sweep.csv"
    if not csv_path.exists():
        print(f"no CSV at {csv_path}; run run_mmin_sweep.py first")
        raise SystemExit(1)
    rows = list(csv.DictReader(open(csv_path)))

    refs = {r["method"]: r for r in rows if r["method"] != "ours"}
    ours = [r for r in rows if r["method"] == "ours"]
    ours.sort(key=lambda r: int(r["m_min"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: total time vs m_min
    xs = [int(r["m_min"]) for r in ours if fnum(r["total_time"]) is not None]
    ys = [fnum(r["total_time"]) for r in ours if fnum(r["total_time"]) is not None]
    ax1.plot(xs, ys, "-o", color="#378ADD", linewidth=2, markersize=6,
             label="Our framework (vary m_min)", zorder=3)
    for m, r in [(m, r) for m in [int(x["m_min"]) for x in ours] for r in [None]]:
        pass
    # reference horizontal lines
    for meth, r in refs.items():
        t = fnum(r["total_time"])
        if t is None:
            continue
        ax1.axhline(t, color=REF_COLORS.get(meth, "#444"), linestyle="--",
                    linewidth=1.5, label=f"{REF_LABELS.get(meth, meth)} ({t:.0f})")
    # mark the cliff (m_min values that failed)
    failed = [int(r["m_min"]) for r in ours if fnum(r["total_time"]) is None]
    if failed:
        ax1.axvspan(min(failed) - 0.5, max(failed) + 0.5, color="#E24B4A", alpha=0.08)
        ax1.text(min(failed), ax1.get_ylim()[1] * 0.95, "accuracy cliff\n(target never reached)",
                 fontsize=8, color="#A32D2D", va="top")
    ax1.set_xlabel("m_min  (privacy / suppression strength →)")
    ax1.set_ylabel(f"Total straggler-aware time to target acc")
    ax1.set_title("Privacy knob vs total time-to-accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # right: rounds-to-accuracy for all frameworks
    labels, vals, colors = [], [], []
    for meth in ("fedavg", "tifl", "tifl_adaptive"):
        if meth in refs and fnum(refs[meth]["rounds"]) is not None:
            labels.append(REF_LABELS[meth])
            vals.append(fnum(refs[meth]["rounds"]))
            colors.append(REF_COLORS[meth])
    for r in ours:
        rd = fnum(r["rounds"])
        if rd is not None:
            labels.append(f"ours m={r['m_min']}")
            vals.append(rd)
            colors.append("#378ADD")
    ax2.bar(range(len(labels)), vals, color=colors)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Rounds to reach target acc")
    ax2.set_title("Rounds-to-accuracy by framework")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "mmin_sweep.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
