"""Step 1 go/no-go gate run. The milestone the whole plan drives toward.

Builds the L1 capability-inference attacker against (a) an undefended run and
(b) a strongest-defended run, then evaluates Gates 2-4.

Harness role: this script lives OUTSIDE the dtfl package because it legitimately
reads latent logs to (1) reconstruct the per-device observed tier sequences an
attacker watching those devices would obtain, and (2) evaluate predictions
against ground truth. The attacker itself only ever receives observations +
transcript + seed labels.
"""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from dtfl.latent import LatentConfig
from dtfl.sim import Engine, EngineConfig, RoundConfig, SimulationOutput, evaluate_gates
from dtfl.transcript import single_bucket, uniform_width_bucketer
from dtfl.attack import L1FewShotAttacker, ObservationFeaturizer, build_observations
from dtfl.metrics import capability_advantage


def device_tier_records(out: SimulationOutput) -> dict[int, list[tuple[int,int]]]:
    """Reconstruct each device's observed (round, tier) sequence.

    An attacker watching a device learns that device's tier ONLY when that
    tier-round actually released an aggregate -- a suppressed tier produces no
    visible release event the observer can pin the device to. So we filter the
    device->tier membership through the TRANSCRIPT's released set. This is what
    makes m_min suppression and (downstream) bucketing genuinely degrade the
    per-device signal, rather than the attacker bypassing defenses by reading
    ground-truth tiers.
    """
    view = out.transcript.view()
    released = {(r.round_index, r.tier_index) for r in view.released()}
    recs: dict[int, list[tuple[int,int]]] = {}
    for log in out.latent_logs:
        for k, ids in enumerate(log.active_device_ids):
            if (log.round_index, k) not in released:
                continue  # suppressed: no observable release to attribute
            for did in ids:
                recs.setdefault(int(did), []).append((log.round_index, k))
    return recs


def make_advantage_fn(seed_frac: float, num_rounds: int, seed: int):
    """Return an advantage_fn(SimulationOutput)->float for the gate.

    Trains L1 on a labeled seed fraction, evaluates advantage on the rest.
    num_tiers is inferred per-run from the transcript (the defended run may use
    a different K than the undefended one).
    """
    def advantage_fn(out: SimulationOutput) -> float:
        recs = device_tier_records(out)
        if not recs:
            return 0.0  # everything suppressed: attacker has nothing -> no advantage
        observations = build_observations(recs)
        ids = np.array([o.device_id for o in observations])
        labels = out.true_classes[ids]

        # Infer K from the transcript's deadline vectors.
        num_tiers = max(r.deadlines.num_tiers for r in out.transcript.view().all())

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(observations))
        n_seed = max(2, int(seed_frac * len(observations)))
        seed_idx, query_idx = perm[:n_seed], perm[n_seed:]
        if len(query_idx) == 0:
            return 0.0

        featurizer = ObservationFeaturizer(num_tiers, num_rounds, view=out.transcript.view())
        atk = L1FewShotAttacker(featurizer, min_observations=5, random_state=seed)
        atk.fit([observations[i] for i in seed_idx], labels[seed_idx])

        query_obs = [observations[i] for i in query_idx]
        y_pred = atk.predict(query_obs)
        y_true = labels[query_idx]
        return capability_advantage(y_true, y_pred).advantage
    return advantage_fn


def main():
    import config as C
    NUM_DEVICES = C.DEVICES
    NUM_ROUNDS = C.PRIVACY_ROUNDS
    K = C.TIERS  # match the rest of the privacy experiments (K=5)
    lcfg = LatentConfig()  # default eta=0.20

    print("="*64)
    print("STEP 1 GO/NO-GO GATE")
    print(f"  N={NUM_DEVICES} rounds={NUM_ROUNDS} K={K} eta={lcfg.proxy_noise_eta} "
          f"SNR={lcfg.signal_to_noise:.2f}")
    print("="*64)

    # --- Undefended run: few tiers, tiny m_min, fine timing buckets (max signal) ---
    eng_undef = Engine(
        EngineConfig(seed=101, num_devices=NUM_DEVICES, num_rounds=NUM_ROUNDS,
                     round_config=RoundConfig(m_min=8),
                     flhetbench=C.population_config()),
        lcfg,
    )
    # K interior quantiles at k/K for k=1..K-1 (calibrate appends a covering tier),
    # so the undefended run uses the same tier count as the rest of the paper.
    quantiles_undef = tuple(k / K for k in range(1, K))
    cuts = eng_undef.calibrate_fixed_deadlines(quantiles_undef)
    round_budget = cuts[-1]
    fine_bucketer = uniform_width_bucketer(num_buckets=20, round_budget=round_budget)
    out_undef = eng_undef.run(lambda r,v: cuts, bucketer=fine_bucketer)

    # --- Defended run: MANY tiers (so fast/sparse tiers fall below m_min and get
    #     suppressed), large m_min, single coarse bucket (min attacker signal) ---
    K_DEF = 8
    quantiles_def = tuple(np.linspace(1/K_DEF, (K_DEF-1)/K_DEF, K_DEF-1))
    eng_def = Engine(
        EngineConfig(seed=101, num_devices=NUM_DEVICES, num_rounds=NUM_ROUNDS,
                     round_config=RoundConfig(m_min=120),
                     flhetbench=C.population_config()),
        lcfg,
    )
    cuts_d = eng_def.calibrate_fixed_deadlines(quantiles_def)
    out_def = eng_def.run(lambda r,v: cuts_d, bucketer=single_bucket)

    advantage_fn = make_advantage_fn(seed_frac=0.10,
                                     num_rounds=NUM_ROUNDS, seed=0)

    report = evaluate_gates(
        undefended_output=out_undef,
        defended_output=out_def,
        advantage_fn=advantage_fn,
        snr=lcfg.signal_to_noise,
    )
    print(report.summary())
    print("="*64)
    return report


if __name__ == "__main__":
    main()