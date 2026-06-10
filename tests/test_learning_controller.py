"""Learning-layer and controller tests."""

from __future__ import annotations

import numpy as np
import pytest

from dtfl.controller import (
    EWMAQuantileController,
    FixedEqualWidth,
    FixedQuantile,
    QuantileTrackingController,
)
from dtfl.latent import LatentConfig
from dtfl.learning import (
    FedTrainConfig,
    NumpySoftmaxModel,
    federated_train,
    iid_shard,
    make_synthetic_classification,
)
from dtfl.protocol import TierContribution, size_weighted_merge
from dtfl.rng import RngHub
from dtfl.sim import Engine, EngineConfig, RoundConfig


def test_fedavg_equivalence_with_real_deltas():
    """Merge of real local-SGD deltas == mean delta over all participants."""
    hub = RngHub(seed=1)
    train, _ = make_synthetic_classification(4000, 20, 5, hub.stream("data"), separation=1.0)
    model = NumpySoftmaxModel(20, 5, seed=0)
    w0 = model.get_params()
    shards = iid_shard(train, 30, hub.stream("shard"))
    deltas = []
    for i in range(30):
        model.set_params(w0)
        deltas.append(model.local_update(shards[i], 1, 0.5, 32, np.random.default_rng(1000 + i)))
    deltas = np.array(deltas)
    SA, SB = deltas[:18].sum(0), deltas[18:].sum(0)
    merged = size_weighted_merge([TierContribution(SA, 18), TierContribution(SB, 12)])
    assert np.allclose(merged, deltas.mean(0), atol=1e-12)


def test_mlp_learning_curve_improves():
    """The nonconvex MLP shows a genuine multi-round learning curve."""
    hub = RngHub(seed=1)
    train, val = make_synthetic_classification(8000, 30, 8, hub.stream("data"), separation=1.4)
    lcfg = LatentConfig()
    eng = Engine(
        EngineConfig(seed=42, num_devices=300, num_rounds=60,
                     round_config=RoundConfig(m_min=8)),
        lcfg,
    )
    cuts = eng.calibrate_fixed_deadlines((0.33, 0.66))
    model = NumpySoftmaxModel(30, 8, hidden_dim=64, seed=0)
    res = federated_train(
        model, train, val, eng, cuts,
        FedTrainConfig(local_epochs=1, local_lr=0.05, server_lr=1.0),
        RoundConfig(m_min=8), base_seed=7,
    )
    assert len(res.val_accuracy) == 60
    assert max(res.val_accuracy) > res.val_accuracy[0] + 0.1
    assert res.round_to_accuracy(0.4) is not None


@pytest.fixture
def controller_setup():
    lcfg = LatentConfig()
    eng = Engine(
        EngineConfig(seed=42, num_devices=2000, num_rounds=50,
                     round_config=RoundConfig(m_min=8)),
        lcfg,
    )
    pilot = eng.calibrate_fixed_deadlines((0.33, 0.66))
    return lcfg, eng, pilot


def _last_cdf(out, K):
    v = out.transcript.view()
    last = max(v.rounds())
    c = [v.tier_counts(last).get(k) or 0 for k in range(K)]
    tot = sum(c)
    return np.cumsum(c) / tot if tot > 0 else np.zeros(K)


def test_quantile_controller_tracks_targets(controller_setup):
    lcfg, eng, pilot = controller_setup
    budget = pilot[-1]
    ctrl = QuantileTrackingController(
        3, (0.33, 0.66, 1.0), tuple(budget * (k + 1) / 3 for k in range(3))
    )
    out = eng.run(ctrl.policy())
    cdf = _last_cdf(out, 3)
    assert abs(cdf[0] - 0.33) + abs(cdf[1] - 0.66) < 0.25, cdf


def test_fixed_quantile_runs(controller_setup):
    lcfg, eng, pilot = controller_setup
    out = eng.run(FixedQuantile(3, pilot).policy())
    assert len(out.transcript.view()) > 0


def test_ewma_beats_fixed_under_drift():
    """Under drift, EWMA tracks targets better than a frozen fixed policy."""
    from dtfl.latent import DriftConfig

    lcfg = LatentConfig(
        drift=DriftConfig(regime_shift_prob=0.05, regime_shift_scale=0.4, slow_drift_per_round=0.02)
    )
    eng0 = Engine(
        EngineConfig(seed=42, num_devices=2000, num_rounds=60,
                     round_config=RoundConfig(m_min=8)),
        LatentConfig(),
    )
    pilot = eng0.calibrate_fixed_deadlines((0.33, 0.66))
    budget = pilot[-1]

    eng_e = Engine(EngineConfig(seed=42, num_devices=2000, num_rounds=60,
                                round_config=RoundConfig(m_min=8)), lcfg)
    ewma = EWMAQuantileController(3, (0.33, 0.66, 1.0),
                                  tuple(budget * (k + 1) / 3 for k in range(3)))
    out_e = eng_e.run(ewma.policy())

    eng_f = Engine(EngineConfig(seed=42, num_devices=2000, num_rounds=60,
                                round_config=RoundConfig(m_min=8)), lcfg)
    out_f = eng_f.run(FixedQuantile(3, pilot).policy())

    cdf_e, cdf_f = _last_cdf(out_e, 3), _last_cdf(out_f, 3)
    err_e = abs(cdf_e[0] - 0.33) + abs(cdf_e[1] - 0.66)
    err_f = abs(cdf_f[0] - 0.33) + abs(cdf_f[1] - 0.66)
    assert err_e <= err_f + 0.05, (err_e, err_f)
