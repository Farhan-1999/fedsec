"""Linkability / anonymity horizon experiment.

Computes the L_i(h)-vs-horizon curve from real runs. Unlike capability inference
(where timing buckets were found inert because tier identity carries the class
signal), linkability SHOULD be sensitive to bucket granularity: finer release-
time buckets make signatures more distinct, shrinking anonymity sets. We compare
finest vs coarsest timing to show this.

Builds the per-device per-round (tier, bucket) atoms from the released-tier-
filtered observations -- the same visible channel the attacker uses -- and never
reads capability class (anonymity is a property of the visible signature alone).
"""
from __future__ import annotations

import config as C

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dtfl.defense import BucketMode, DefenseConfig, bucketer_for, deadline_quantiles_for
from dtfl.latent import LatentConfig
from dtfl.metrics import linkability_curve
from dtfl.sim import Engine, EngineConfig, RoundConfig

OUT_CSV = Path("artifacts/results/linkability.csv")


def per_round_atoms(out, m_min_irrelevant=None):
    """Build device_id -> {round -> (tier, bucket)} from released tier-rounds.

    The bucket is read from the transcript record (the visible coarse bucket),
    NOT from any latent timing. A device appears for a round only if its tier
    that round released.
    """
    view = out.transcript.view()
    # (round, tier) -> bucket, for released tiers
    bucket_of = {
        (r.round_index, r.tier_index): r.release_bucket
        for r in view.released()
    }
    released = set(bucket_of.keys())
    atoms: dict[int, dict[int, tuple[int, int]]] = {}
    for log in out.latent_logs:
        for k, ids in enumerate(log.active_device_ids):
            key = (log.round_index, k)
            if key not in released:
                continue
            b = bucket_of[key]
            for did in ids:
                atoms.setdefault(int(did), {})[log.round_index] = (k, int(b) if b is not None else -1)
    return atoms


def run_for_bucketmode(label, bucket_kwargs, lcfg, m_min=1, num_tiers=5,
                       seed=101, num_devices=C.DEVICES, num_rounds=C.PRIVACY_ROUNDS):
    defense = DefenseConfig(m_min=m_min, num_tiers=num_tiers, **bucket_kwargs)
    eng = Engine(EngineConfig(seed=seed, num_devices=num_devices, num_rounds=num_rounds,
                              round_config=RoundConfig(m_min=m_min)), lcfg)
    cuts = eng.calibrate_fixed_deadlines(deadline_quantiles_for(defense))
    bucketer = bucketer_for(defense, round_budget=cuts[-1])
    out = eng.run(lambda r, v: cuts, bucketer=bucketer)
    atoms = per_round_atoms(out)
    horizons = [1, 2, 5, 10, 20, 40, 80]
    curve = linkability_curve(atoms, horizons, m_min=m_min)
    return curve


def main():
    lcfg = LatentConfig()
    print("=" * 72)
    print(f"LINKABILITY / ANONYMITY  (eta={lcfg.proxy_noise_eta}, SNR={lcfg.signal_to_noise:.2f})")
    print("=" * 72)
    print("Finding: linkability is driven by the TIER SEQUENCE (the same channel")
    print("as capability inference); release-time buckets are redundant with tier")
    print("and do not change the curve. The contrast that matters is tier count K:")
    print("more tiers => more distinct sequences => faster anonymity collapse.\n")

    # Contrast tier count K (the real linkability lever), buckets held coarse.
    configs = [
        ("K=3", dict(bucket_mode=BucketMode.SINGLE), 3),
        ("K=5", dict(bucket_mode=BucketMode.SINGLE), 5),
        ("K=8", dict(bucket_mode=BucketMode.SINGLE), 8),
    ]

    rows = []
    for label, bkw, K in configs:
        print(f"--- {label} ---")
        curve = run_for_bucketmode(label, bkw, lcfg, num_tiers=K)
        print(curve.summary())
        print()
        for i, h in enumerate(curve.horizons):
            rows.append({
                "config": label,
                "num_tiers": K,
                "horizon": h,
                "mean_linkability": curve.mean_linkability[i],
                "median_anon_set": curve.median_anon_set[i],
                "fraction_unique": curve.fraction_unique[i],
                "fraction_below_m_min": curve.fraction_below_m_min[i],
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("=" * 72)
    print(f"wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
