"""Step 3: the adversary ladder L0 <= L1 <= L3(omniscient).

Runs three rungs against the SAME transcripts at a few defense points, so the
realistic L1 number is bracketed by an unsupervised floor (L0) and an omniscient
ceiling. Reporting the ladder is what converts "an attack works" into "we
characterized leakage across adversary strength".

All rungs share the identical observation channel (the same featurizer over the
same released-tier-filtered observations); they differ ONLY in auxiliary
knowledge:
  L0: none (cluster + tier-order labeling)
  L1: labeled seed (10%)
  L3: all-other-device labels (membership-inference ceiling)

NOTE ON NAMING: an older Bayes-optimal rung (then called L2, "knows the
generative model") has been REMOVED -- the server broadcasts the deadlines and
observes the count distribution, so that rung's "secret" generative model is
largely recoverable from aggregates and it was not cleanly distinct. It is no
longer computed here. The emitted CSV keeps the legacy column name ``L3`` for the
omniscient rung so existing plotting scripts keep working; the paper presents
this rung as "L2 (omniscient)". See experiments/plots/plot_ladder.py, which maps
CSV ``L3`` -> paper ``L2``.
"""
from __future__ import annotations

import config as C

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import quiet  # noqa: F401  (installs warning filters on import)
sys.path.insert(0, str(Path(__file__).parent))

from harness import _device_tier_records  # reuse the released-filtered observation builder

from dtfl.attack import (
    L0UnsupervisedAttacker,
    L1FewShotAttacker,
    L3OmniscientAttacker,
    ObservationFeaturizer,
    build_observations,
)
from dtfl.defense import DefenseConfig, bucketer_for, deadline_quantiles_for
from dtfl.latent import LatentConfig
from dtfl.metrics import capability_advantage
from dtfl.sim import Engine, EngineConfig, RoundConfig

OUT_CSV = Path("artifacts/results/ladder.csv")


def run_one_defense(defense: DefenseConfig, lcfg: LatentConfig, *,
                    seed=101, num_devices=C.DEVICES, num_rounds=C.PRIVACY_ROUNDS, seed_frac=0.10, attack_seed=0):
    """Run the simulation once, then evaluate all four rungs on its transcript."""
    engine = Engine(
        EngineConfig(seed=seed, num_devices=num_devices, num_rounds=num_rounds,
                     round_config=RoundConfig(m_min=defense.m_min),
                     flhetbench=C.population_config()),
        lcfg,
    )
    quantiles = deadline_quantiles_for(defense)
    controller = engine.adaptive_policy(quantiles)
    bucketer = bucketer_for(defense, round_budget=controller._cutoffs[-1])
    out = engine.run(controller.policy(), bucketer=bucketer)

    recs = _device_tier_records(out)
    observations = build_observations(recs)
    if not observations:
        return None
    ids = np.array([o.device_id for o in observations])
    labels = out.true_classes[ids]
    K = max(r.deadlines.num_tiers for r in out.transcript.view().all())
    featurizer = ObservationFeaturizer(K, num_rounds, view=out.transcript.view())

    # Common train/query split (used by L1; L0 ignores labels; L3 all-but-target).
    rng = np.random.default_rng(attack_seed)
    perm = rng.permutation(len(observations))
    n_seed = max(2, int(seed_frac * len(observations)))
    seed_idx, query_idx = perm[:n_seed], perm[n_seed:]
    query_obs = [observations[i] for i in query_idx]
    y_true = labels[query_idx]

    results = {}

    # --- L0: cluster everything, predict the query set ---
    l0 = L0UnsupervisedAttacker(featurizer, num_classes=lcfg.num_classes, random_state=attack_seed)
    l0.fit(observations)  # clusters on all observed devices, labels ignored
    results["L0"] = capability_advantage(y_true, l0.predict(query_obs)).advantage

    # --- L1: labeled seed ---
    l1 = L1FewShotAttacker(featurizer, random_state=attack_seed)
    l1.fit([observations[i] for i in seed_idx], labels[seed_idx])
    results["L1"] = capability_advantage(y_true, l1.predict(query_obs)).advantage

    # --- L3: all-other-device labels (ceiling; presented as "L2 omniscient") ---
    l3 = L3OmniscientAttacker(featurizer, random_state=attack_seed)
    l3.fit(observations, labels)  # full omniscient pool
    results["L3"] = capability_advantage(y_true, l3.predict(query_obs)).advantage

    return results


def main():
    lcfg = LatentConfig()
    # Three defense points spanning the frontier: undefended, mid (where the
    # rungs should separate most), and strong.
    points = [
        ("undefended", DefenseConfig(m_min=1, num_tiers=5)),
        ("mid",        DefenseConfig(m_min=77, num_tiers=5)),
        ("strong",     DefenseConfig(m_min=88, num_tiers=5)),
    ]
    print("=" * 64)
    print(f"STEP 3 ADVERSARY LADDER  (eta={lcfg.proxy_noise_eta}, SNR={lcfg.signal_to_noise:.2f})")
    print("=" * 64)
    print(f"{'defense':>12} {'L0':>7} {'L1':>7} {'L3':>7}")
    print(f"{'':>12} {'':>7} {'':>7} {'(paper L2)':>7}")
    print("-" * 64)
    rows = []
    for name, defense in points:
        res = run_one_defense(defense, lcfg)
        if res is None:
            print(f"{name:>12}   (all suppressed)")
            continue
        print(f"{name:>12} {res['L0']:>7.3f} {res['L1']:>7.3f} {res['L3']:>7.3f}")
        rows.append({"defense": name, "m_min": defense.m_min, **res})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("-" * 64)
    print(f"wrote {len(rows)} points -> {OUT_CSV}")


if __name__ == "__main__":
    main()