"""Multi-seed hardening of the adversary ladder and the linkability curve.

Reuses the generic run_multiseed driver to add 95% CIs to:
  (1) the four-rung ladder (L0 <= L1 ~ L2 <= L3) at the mid-frontier defense
      point, where the rungs separate and variance matters most, and
  (2) the linkability horizon curve at selected horizons.

Smaller N and fewer rounds than the single-seed runs so S seeds finish in
reasonable time; the qualitative structure (ladder ordering, anonymity collapse)
is what we are confirming is seed-stable, plus the per-point CIs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from multiseed import run_multiseed
from run_ladder import run_one_defense
from run_linkability import run_for_bucketmode

import config as C
from dtfl.defense import BucketMode, DefenseConfig
from dtfl.latent import LatentConfig

OUT_LADDER = Path("artifacts/results/multiseed_ladder.csv")
OUT_LINK = Path("artifacts/results/multiseed_linkability.csv")

NUM_SEEDS = C.NUM_SEEDS
NUM_DEVICES = C.DEVICES
NUM_ROUNDS = C.PRIVACY_ROUNDS


def harden_ladder(lcfg: LatentConfig):
    """Ladder at three defense points, each across S seeds."""
    print("=" * 76)
    print("MULTI-SEED LADDER")
    print("=" * 76)
    # cliff points chosen for N=1500 (per the frontier probe: cliff ~ m_min 45-65).
    points = [
        ("undefended", DefenseConfig(m_min=1, num_tiers=5)),
        ("mid", DefenseConfig(m_min=50, num_tiers=5)),
        ("strong", DefenseConfig(m_min=55, num_tiers=5)),
    ]
    rows = []
    for name, defense in points:
        print(f"\n--- {name} ({defense.describe()}) ---")

        def eval_fn(engine_seed: int, attack_seed: int, _d=defense):
            res = run_one_defense(
                _d, lcfg,
                seed=engine_seed, attack_seed=attack_seed,
                num_devices=NUM_DEVICES, num_rounds=NUM_ROUNDS,
            )
            # res is {"L0":..,"L1":..,"L2":..,"L3":..} or None if all suppressed.
            return res if res is not None else {"L0": 0.0, "L1": 0.0, "L2": 0.0, "L3": 0.0}

        agg = run_multiseed(eval_fn, num_seeds=NUM_SEEDS, base_seed=2024)
        row = {"point": name, "m_min": defense.m_min}
        for rung in ("L0", "L1", "L2", "L3"):
            a = agg.aggregates[rung]
            print("  " + a.summary())
            row[f"{rung}_mean"] = a.mean
            row[f"{rung}_ci_half"] = a.ci_halfwidth
        rows.append(row)

    OUT_LADDER.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LADDER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote ladder -> {OUT_LADDER}")


def harden_linkability(lcfg: LatentConfig):
    """Linkability curve (K=5) across S seeds, at selected horizons."""
    print("\n" + "=" * 76)
    print("MULTI-SEED LINKABILITY (K=5)")
    print("=" * 76)
    track_horizons = [5, 10, 20, 40]

    def eval_fn(engine_seed: int, attack_seed: int):
        # linkability uses only the engine seed (no attack split).
        curve = run_for_bucketmode(
            "single", dict(bucket_mode=BucketMode.SINGLE), lcfg,
            m_min=1, num_tiers=5,
            seed=engine_seed, num_devices=NUM_DEVICES, num_rounds=NUM_ROUNDS,
        )
        out = {}
        for h in track_horizons:
            idx = curve.horizons.index(h)
            out[f"uniq_h{h}"] = curve.fraction_unique[idx]
            out[f"meanL_h{h}"] = curve.mean_linkability[idx]
        out["accum_rate"] = curve.accumulation_rate
        return out

    agg = run_multiseed(eval_fn, num_seeds=NUM_SEEDS, base_seed=2024)
    rows = []
    for h in track_horizons:
        u = agg.aggregates[f"uniq_h{h}"]
        ml = agg.aggregates[f"meanL_h{h}"]
        print(f"  h={h:>3}: unique {100*u.mean:.1f}% +/- {100*u.ci_halfwidth:.1f}  "
              f"| meanL {ml.mean:.3f} +/- {ml.ci_halfwidth:.3f}")
        rows.append({
            "horizon": h,
            "uniq_mean": u.mean, "uniq_ci_half": u.ci_halfwidth,
            "meanL_mean": ml.mean, "meanL_ci_half": ml.ci_halfwidth,
        })
    ar = agg.aggregates["accum_rate"]
    print("  " + ar.summary())

    OUT_LINK.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LINK, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote linkability -> {OUT_LINK}")


def main():
    lcfg = LatentConfig()
    print(f"S={NUM_SEEDS} N={NUM_DEVICES} rounds={NUM_ROUNDS} "
          f"eta={lcfg.proxy_noise_eta} SNR={lcfg.signal_to_noise:.2f}")
    harden_ladder(lcfg)
    harden_linkability(lcfg)


if __name__ == "__main__":
    main()
