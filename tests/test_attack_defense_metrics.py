"""Attack, defense, metrics, and gate tests.

Covers the L1 attack producing nonzero advantage (Gate 2 precondition), the
defense knobs each having a measurable effect, the advantage and linkability
metrics, and the anonymity-collapse direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from dtfl.attack import (
    L1FewShotAttacker,
    ObservationFeaturizer,
    build_observations,
)
from dtfl.defense import (
    BucketMode,
    CountMode,
    DefenseConfig,
    apply_count_defense,
    apply_padding,
    apply_send_delay,
    bucketer_for,
    deadline_quantiles_for,
)
from dtfl.metrics import (
    capability_advantage,
    equivalence_class_sizes,
    linkability_curve,
    signature_up_to,
)


# ---------------- defense knobs ----------------

def test_count_modes():
    rng = np.random.default_rng(0)
    assert apply_count_defense(57, DefenseConfig(count_mode=CountMode.EXACT), rng) == 57
    assert apply_count_defense(57, DefenseConfig(count_mode=CountMode.HIDDEN), rng) is None
    assert apply_count_defense(57, DefenseConfig(count_mode=CountMode.ROUNDED, count_round_to=10), rng) == 60


def test_send_delay_only_delays():
    rng = np.random.default_rng(0)
    tau = np.array([1.0, 2.0, 3.0])
    cutoffs = np.array([1.5, 2.5, 5.0])
    jit = apply_send_delay(tau, cutoffs, DefenseConfig(send_delay_scale=0.3), rng)
    assert np.all(jit >= tau) and np.any(jit > tau)


def test_padding_inflates_to_target():
    pad = apply_padding(20, 50)
    assert pad.revealed_count == 50 and pad.num_dummies == 30
    assert apply_padding(80, 50).num_dummies == 0


def test_deadline_quantiles_and_bucketer():
    assert deadline_quantiles_for(DefenseConfig(num_tiers=3)) == (1 / 3, 2 / 3)
    from dtfl.types import RoundDeadlines
    dl = RoundDeadlines(0, (1.0, 2.0, 3.0))
    b = bucketer_for(DefenseConfig(bucket_mode=BucketMode.SINGLE), 3.0)
    assert b(2.9, dl) == 0


# ---------------- advantage metric ----------------

def test_advantage_zero_at_prior():
    y = np.array([0, 0, 0, 1])  # majority class 0, prior 0.75
    res = capability_advantage(y, np.zeros(4, dtype=int))  # predict majority
    assert res.advantage == 0.0


def test_advantage_one_at_perfect():
    y = np.array([0, 1, 2, 3])
    res = capability_advantage(y, y.copy())
    assert res.advantage == pytest.approx(1.0)


# ---------------- L1 attack produces signal (Gate 2 precondition) ----------------

def test_l1_attack_has_advantage():
    """On a clean run with no defenses the L1 attacker beats the prior."""
    from dtfl.latent import LatentConfig
    from dtfl.sim import Engine, EngineConfig, RoundConfig

    lcfg = LatentConfig()
    eng = Engine(
        EngineConfig(seed=101, num_devices=2000, num_rounds=60,
                     round_config=RoundConfig(m_min=8)),
        lcfg,
    )
    cuts = eng.calibrate_fixed_deadlines((0.33, 0.66))
    out = eng.run(lambda r, v: cuts)

    # observed (round, tier) per device, released tiers only
    view = out.transcript.view()
    released = {(r.round_index, r.tier_index) for r in view.released()}
    recs = {}
    for log in out.latent_logs:
        for k, ids in enumerate(log.active_device_ids):
            if (log.round_index, k) in released:
                for did in ids:
                    recs.setdefault(int(did), []).append((log.round_index, k))

    obs = build_observations(recs)
    ids = np.array([o.device_id for o in obs])
    labels = out.true_classes[ids]
    K = max(r.deadlines.num_tiers for r in view.all())

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    n_seed = max(2, len(obs) // 10)
    seed_idx, query_idx = perm[:n_seed], perm[n_seed:]

    feat = ObservationFeaturizer(K, 60, view=view)
    atk = L1FewShotAttacker(feat, random_state=0)
    atk.fit([obs[i] for i in seed_idx], labels[seed_idx])
    pred = atk.predict([obs[i] for i in query_idx])
    adv = capability_advantage(labels[query_idx], pred).advantage
    assert adv > 0.05, f"Gate 2 precondition: undefended advantage should be >0, got {adv}"


# ---------------- linkability / anonymity ----------------

def test_equivalence_classes():
    sigs = {0: ("a",), 1: ("a",), 2: ("b",)}
    sizes = equivalence_class_sizes(sigs)
    assert sizes == {0: 2, 1: 2, 2: 1}


def test_signature_up_to_includes_gaps():
    atoms = {0: (1, 2), 2: (0, 1)}  # round 1 missing
    sig = signature_up_to(atoms, 3)
    assert sig == ((1, 2), (-1, -1), (0, 1))


def test_anonymity_collapses_with_horizon():
    """More observation -> smaller anonymity sets (mean linkability rises)."""
    rng = np.random.default_rng(0)
    # 200 devices, each a random tier sequence over 30 rounds
    atoms = {}
    for d in range(200):
        atoms[d] = {r: (int(rng.integers(0, 5)), 0) for r in range(30)}
    curve = linkability_curve(atoms, [2, 10, 30], m_min=8)
    # mean linkability is non-decreasing in horizon
    assert curve.mean_linkability[0] <= curve.mean_linkability[1] <= curve.mean_linkability[2]
    assert curve.accumulation_rate > 0


# ---------------- secagg cost model ----------------

def test_secagg_calibration_matches_simulator():
    """Analytic release probability matches Monte-Carlo over the release gate."""
    from dtfl.protocol.dropout import DropoutRates, apply_dropout
    from dtfl.protocol.release import decide_release
    from dtfl.protocol.threshold import reconstruction_threshold
    from dtfl.rng import RngHub
    from dtfl.secagg import tier_success_probability

    rates = DropoutRates(rho_mask=0.10, rho_unmask=0.05)
    n, m_min = 40, 32
    t = reconstruction_threshold(n, m_min, rates)
    pred = tier_success_probability(n, m_min, t, rates).prob_success
    hub = RngHub(seed=5)
    trials = 4000
    succ = 0
    for i in range(trials):
        oc = apply_dropout(n, rates, hub.stream(f"t{i}"))
        succ += int(decide_release(oc, m_min, t).released)
    emp = succ / trials
    assert abs(pred - emp) < 0.03, (pred, emp)


def test_sparse_cheaper_than_complete_at_scale():
    """Sparse-graph per-client cost is far below complete-graph for large tiers."""
    from dtfl.secagg import complete_graph_cost, sparse_graph_cost

    comp = complete_graph_cost(1000, 50000)
    sparse = sparse_graph_cost(1000, 50000)
    assert sparse.degree < comp.degree
    assert sparse.setup_latency_sec < comp.setup_latency_sec
    # sparse degree is logarithmic
    assert sparse.degree <= 20
