"""Experiment harness: evaluate one defense point end to end.

Threads a DefenseConfig through the engine, runs the L1 attacker against the
resulting transcript, and returns the privacy outcome (advantage) plus utility
outcomes (participation, suppression, merge bias from count perturbation).

This is the integration the defense layer deferred: it builds deadlines from the
configured tier count, the bucketer from the configured timing resolution, and
passes m_min into the round config. Count/jitter/padding knobs are applied via a
custom round path when active; for the primary m_min x bucket sweep the engine's
standard path suffices (those two knobs route through deadlines + RoundConfig +
bucketer with no per-round modification).

Harness role (lives outside the package): it reads latent logs to reconstruct
the attacker's observable inputs and to evaluate against ground truth. The
attacker itself receives only observations + transcript + seed labels.
"""
from __future__ import annotations

from pathlib import Path

import config as C

import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtfl.attack import L1FewShotAttacker, ObservationFeaturizer, build_observations
from dtfl.defense import DefenseConfig, bucketer_for, deadline_quantiles_for
from dtfl.latent import LatentConfig
from dtfl.metrics import capability_advantage
from dtfl.sim import Engine, EngineConfig, RoundConfig, SimulationOutput, snr_diagnostic

__all__ = ["DefensePointResult", "evaluate_defense_point"]


@dataclass
class DefensePointResult:
    """Outcome of one defense configuration."""

    # defense identity
    m_min: int
    num_tiers: int
    bucket_label: str
    # privacy
    advantage: float
    accuracy: float
    prior: float
    class_tier_mi: float  # ground-truth leakage diagnostic
    # utility
    mean_participation: float
    release_rate: float  # fraction of tier-rounds that released
    # bookkeeping
    n_query: int

    def row(self) -> dict:
        return {
            "m_min": self.m_min,
            "num_tiers": self.num_tiers,
            "bucket": self.bucket_label,
            "advantage": self.advantage,
            "accuracy": self.accuracy,
            "prior": self.prior,
            "class_tier_mi": self.class_tier_mi,
            "mean_participation": self.mean_participation,
            "release_rate": self.release_rate,
            "n_query": self.n_query,
        }


def _device_tier_records(out: SimulationOutput) -> dict[int, list[tuple[int, int]]]:
    """Observed (round, tier) per device, filtered to RELEASED tier-rounds.

    An attacker watching a device attributes its tier only when that tier-round
    produced a visible release; suppression therefore removes signal. This is the
    load-bearing adversary-observation model (confirmed with the user).
    """
    view = out.transcript.view()
    released = {(r.round_index, r.tier_index) for r in view.released()}
    recs: dict[int, list[tuple[int, int]]] = {}
    for log in out.latent_logs:
        for k, ids in enumerate(log.active_device_ids):
            if (log.round_index, k) not in released:
                continue
            for did in ids:
                recs.setdefault(int(did), []).append((log.round_index, k))
    return recs


def _release_rate(out: SimulationOutput) -> float:
    view = out.transcript.view()
    total = len(view.all())
    if total == 0:
        return 0.0
    return len(view.released()) / total


def evaluate_defense_point(
    defense: DefenseConfig,
    latent_config: LatentConfig,
    *,
    seed: int = 101,
    num_devices: int = C.DEVICES,
    num_rounds: int = C.PRIVACY_ROUNDS,
    seed_frac: float = 0.10,
    attack_seed: int = 0,
) -> DefensePointResult:
    """Run the simulation under ``defense`` and evaluate the L1 attack on it."""
    engine = Engine(
        EngineConfig(
            seed=seed,
            num_devices=num_devices,
            num_rounds=num_rounds,
            round_config=RoundConfig(m_min=defense.m_min),
        ),
        latent_config,
    )

    # Deadlines for the configured number of tiers, via latent calibration.
    quantiles = deadline_quantiles_for(defense)
    cuts = engine.calibrate_fixed_deadlines(quantiles)
    round_budget = cuts[-1]
    bucketer = bucketer_for(defense, round_budget=round_budget)

    out = engine.run(lambda r, v: cuts, bucketer=bucketer)

    # --- attack ---
    recs = _device_tier_records(out)
    if not recs:
        # Everything suppressed: attacker has no signal -> zero advantage.
        mi, _ = snr_diagnostic(out)
        return DefensePointResult(
            m_min=defense.m_min, num_tiers=defense.num_tiers,
            bucket_label=defense.bucket_mode.value
            + (f"/{defense.num_buckets}" if defense.bucket_mode.value == "uniform" else ""),
            advantage=0.0, accuracy=0.0, prior=0.0, class_tier_mi=mi,
            mean_participation=float(np.mean(out.participation)),
            release_rate=_release_rate(out), n_query=0,
        )

    observations = build_observations(recs)
    ids = np.array([o.device_id for o in observations])
    labels = out.true_classes[ids]
    num_tiers = max(r.deadlines.num_tiers for r in out.transcript.view().all())

    rng = np.random.default_rng(attack_seed)
    perm = rng.permutation(len(observations))
    n_seed = max(2, int(seed_frac * len(observations)))
    seed_idx, query_idx = perm[:n_seed], perm[n_seed:]

    featurizer = ObservationFeaturizer(num_tiers, num_rounds, view=out.transcript.view())
    atk = L1FewShotAttacker(featurizer, min_observations=5, random_state=attack_seed)
    atk.fit([observations[i] for i in seed_idx], labels[seed_idx])

    query_obs = [observations[i] for i in query_idx]
    y_pred = atk.predict(query_obs)
    y_true = labels[query_idx]
    adv = capability_advantage(y_true, y_pred)
    mi, _ = snr_diagnostic(out)

    bucket_label = defense.bucket_mode.value + (
        f"/{defense.num_buckets}" if defense.bucket_mode.value == "uniform" else ""
    )
    return DefensePointResult(
        m_min=defense.m_min,
        num_tiers=defense.num_tiers,
        bucket_label=bucket_label,
        advantage=adv.advantage,
        accuracy=adv.accuracy,
        prior=adv.majority_prior,
        class_tier_mi=mi,
        mean_participation=float(np.mean(out.participation)),
        release_rate=_release_rate(out),
        n_query=len(query_idx),
    )
