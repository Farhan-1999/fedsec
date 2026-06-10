"""Multi-seed hardening of the headline privacy-utility frontier.

Re-runs the key frontier points across S random deployments and reports mean
advantage and mean participation with 95% CIs. The cliff region (where some
tiers clear m_min and some do not) is where variance is expected to be largest --
small population shifts move the transition point -- so these are exactly the
points that need error bars before publication.

Outputs a CSV with per-point mean/CI for both axes, ready for an errorbar plot.
"""
from __future__ import annotations

import config as C

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from harness import evaluate_defense_point
from multiseed import run_multiseed

from dtfl.defense import BucketMode, DefenseConfig
from dtfl.latent import LatentConfig

OUT_CSV = Path("artifacts/results/multiseed_frontier.csv")

NUM_SEEDS = C.NUM_SEEDS
# Smaller population than the single-seed sweep so 5 seeds run in reasonable time;
# the cliff location scales with per-tier mass, so m_min values are chosen for
# this N (1500 devices, ~K tiers). Keep N fixed across points for comparability.
NUM_DEVICES = C.DEVICES
NUM_ROUNDS = C.PRIVACY_ROUNDS


def frontier_points() -> list[tuple[str, DefenseConfig]]:
    """Points spanning the frontier at K=5 for N=1500 (per-tier mass ~ 1500*avail/5)."""
    # With ~270 participants/round over 5 tiers, per-tier mass ~ 54; cliff near there.
    return [
        ("undefended", DefenseConfig(m_min=1, num_tiers=5, bucket_mode=BucketMode.SINGLE)),
        ("mild",       DefenseConfig(m_min=45, num_tiers=5, bucket_mode=BucketMode.SINGLE)),
        ("cliff_lo",   DefenseConfig(m_min=50, num_tiers=5, bucket_mode=BucketMode.SINGLE)),
        ("cliff_mid",  DefenseConfig(m_min=55, num_tiers=5, bucket_mode=BucketMode.SINGLE)),
        ("cliff_hi",   DefenseConfig(m_min=60, num_tiers=5, bucket_mode=BucketMode.SINGLE)),
        ("strong",     DefenseConfig(m_min=70, num_tiers=5, bucket_mode=BucketMode.SINGLE)),
    ]


def main():
    lcfg = LatentConfig()
    print("=" * 76)
    print(f"MULTI-SEED FRONTIER  S={NUM_SEEDS} seeds  N={NUM_DEVICES} rounds={NUM_ROUNDS}")
    print(f"  eta={lcfg.proxy_noise_eta} SNR={lcfg.signal_to_noise:.2f}")
    print("=" * 76)

    rows = []
    for name, defense in frontier_points():
        print(f"\n--- {name}  ({defense.describe()}) ---")

        def eval_fn(engine_seed: int, attack_seed: int, _d=defense):
            res = evaluate_defense_point(
                _d, lcfg,
                seed=engine_seed, attack_seed=attack_seed,
                num_devices=NUM_DEVICES, num_rounds=NUM_ROUNDS,
            )
            return {
                "advantage": res.advantage,
                "participation": res.mean_participation,
                "release_rate": res.release_rate,
            }

        agg = run_multiseed(eval_fn, num_seeds=NUM_SEEDS, base_seed=2024)
        adv = agg.aggregates["advantage"]
        par = agg.aggregates["participation"]
        rel = agg.aggregates["release_rate"]
        print("  " + adv.summary())
        print("  " + par.summary())

        adv_lo, adv_hi = adv.ci95()
        par_lo, par_hi = par.ci95()
        rows.append({
            "point": name,
            "m_min": defense.m_min,
            "adv_mean": adv.mean, "adv_ci_half": adv.ci_halfwidth,
            "adv_lo": adv_lo, "adv_hi": adv_hi,
            "part_mean": par.mean, "part_ci_half": par.ci_halfwidth,
            "part_lo": par_lo, "part_hi": par_hi,
            "rel_mean": rel.mean,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n" + "=" * 76)
    print(f"wrote {len(rows)} points -> {OUT_CSV}")


if __name__ == "__main__":
    main()
