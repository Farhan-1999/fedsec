"""Step 2: the headline m_min x bucket-granularity sweep.

Walks the two PRIMARY defense axes and records, for each point, the L1 attacker's
capability advantage (privacy) against mean participation and release rate
(utility). The output is the data for the headline Pareto figure: advantage vs
time-to-accuracy proxy, each point a defense configuration.

Coupling note: tier count is tied to m_min regime so that suppression genuinely
bites. With ~num_devices * avg_availability participants spread over K tiers,
the per-tier mass must be able to fall below m_min for suppression to matter.
We use more tiers at higher m_min.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from experiments.harness import evaluate_defense_point

from dtfl.defense import BucketMode, DefenseConfig
from dtfl.latent import LatentConfig

OUT_CSV = Path("artifacts/results/headline_pareto.csv")


def bucket_configs() -> list[tuple[str, dict]]:
    """Timing-resolution axis, finest -> coarsest.

    NOTE (finding): the bucket axis turns out to be a WEAK defense -- the
    attacker's signal is dominated by the tier-occupancy histogram, so coarsening
    release-time buckets barely moves advantage. We keep two points (finest and
    single) to document this rather than sweeping it densely.
    """
    return [
        ("uniform/40", dict(bucket_mode=BucketMode.UNIFORM, num_buckets=40)),
        ("single", dict(bucket_mode=BucketMode.SINGLE)),
    ]


def m_min_configs() -> list[tuple[int, int]]:
    """Suppression axis as (m_min, num_tiers) pairs.

    The privacy-utility frontier is a narrow band where SOME tiers clear m_min
    and some do not -- located near the per-tier participant mass. With ~542
    participants this band sits around m_min 80-135 at K=5 and shifts with K.
    We resolve that band finely (the earlier coarse grid stepped over it). The
    transition is sharp ("cliff") because equal-ish tier sizes fall together --
    itself a reported finding.
    """
    pairs = []
    # K=5 band (per-tier mass ~108): resolve 1 -> full suppression.
    for m in [1, 60, 90, 100, 105, 110, 115, 120, 130, 150]:
        pairs.append((m, 5))
    # K=8 band (per-tier mass ~68): cliff sits lower.
    for m in [1, 40, 55, 62, 68, 74, 80, 95]:
        pairs.append((m, 8))
    return pairs


def main():
    lcfg = LatentConfig()  # default eta=0.20, SNR 2.56
    print("=" * 78)
    print(f"STEP 2 HEADLINE SWEEP  (eta={lcfg.proxy_noise_eta}, SNR={lcfg.signal_to_noise:.2f})")
    print("=" * 78)
    header = f"{'m_min':>6} {'K':>3} {'bucket':>12} {'adv':>7} {'acc':>6} {'MI':>7} {'partic':>8} {'rel%':>6}"
    print(header)
    print("-" * 78)

    rows = []
    for (m_min, K) in m_min_configs():
        for (blabel, bkw) in bucket_configs():
            defense = DefenseConfig(m_min=m_min, num_tiers=K, **bkw)
            res = evaluate_defense_point(defense, lcfg)
            rows.append(res.row())
            print(
                f"{res.m_min:>6} {res.num_tiers:>3} {res.bucket_label:>12} "
                f"{res.advantage:>7.3f} {res.accuracy:>6.3f} {res.class_tier_mi:>7.3f} "
                f"{res.mean_participation:>8.1f} {100*res.release_rate:>5.0f}%"
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("-" * 78)
    print(f"wrote {len(rows)} points -> {OUT_CSV}")
    return rows


if __name__ == "__main__":
    main()
