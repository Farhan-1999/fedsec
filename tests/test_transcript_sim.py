"""Transcript and simulation-engine tests, including the determinism property."""

from __future__ import annotations

import numpy as np
import pytest

from dtfl.latent import LatentConfig
from dtfl.sim import Engine, EngineConfig, RoundConfig, snr_diagnostic
from dtfl.transcript import (
    TranscriptStore,
    per_tier_bucket,
    single_bucket,
    uniform_width_bucketer,
)
from dtfl.types import RoundDeadlines, TierFlag, TierRecord


@pytest.fixture
def deadlines():
    return RoundDeadlines(0, (1.0, 2.0, 3.0))


def test_bucketers(deadlines):
    assert single_bucket(2.9, deadlines) == 0
    assert per_tier_bucket(1.5, deadlines) == 1
    ub = uniform_width_bucketer(6, 3.0)
    assert ub(0.25, deadlines) == 0 and ub(2.9, deadlines) == 5
    assert ub(99.0, deadlines) == 5  # clamps overflow


def test_store_view_queries(deadlines):
    store = TranscriptStore()
    store.append(TierRecord(0, 0, TierFlag.RELEASED, 50, 0, np.ones(4), deadlines))
    store.append(TierRecord(0, 1, TierFlag.SUPPRESSED, None, None, None, deadlines))
    v = store.view()
    assert len(v) == 2
    assert len(v.released()) == 1
    assert v.tier_counts(0) == {0: 50, 1: None}
    assert v.signature(0, 0) == (0, 0, "released")


def test_view_reflects_later_appends(deadlines):
    """Online-adversary property: a view sees records appended after issuance."""
    store = TranscriptStore()
    v = store.view()
    store.append(TierRecord(0, 0, TierFlag.RELEASED, 10, 0, np.ones(4), deadlines))
    assert len(v) == 1


@pytest.fixture
def sim_output():
    lcfg = LatentConfig()
    eng = Engine(
        EngineConfig(seed=42, num_devices=1000, num_rounds=40,
                     round_config=RoundConfig(m_min=16)),
        lcfg,
    )
    cuts = eng.calibrate_fixed_deadlines((0.33, 0.66))
    return eng.run(lambda r, v: cuts, bucketer=per_tier_bucket), cuts


def test_simulation_produces_full_transcript(sim_output):
    out, _ = sim_output
    v = out.transcript.view()
    assert v.rounds() == list(range(40))
    assert len(v.released()) > 0


def test_latent_logs_separate_and_aligned(sim_output):
    out, _ = sim_output
    assert len(out.latent_logs) == 40
    assert out.true_classes.shape[0] == 1000


def test_realized_mi_positive(sim_output):
    out, _ = sim_output
    mi, _ = snr_diagnostic(out)
    assert mi > 0.05, mi


def test_determinism_identical_transcript():
    """Same seed -> identical transcript signature (the reproducibility property)."""
    lcfg = LatentConfig()

    def run():
        eng = Engine(
            EngineConfig(seed=42, num_devices=1000, num_rounds=30,
                         round_config=RoundConfig(m_min=16)),
            lcfg,
        )
        cuts = eng.calibrate_fixed_deadlines((0.33, 0.66))
        out = eng.run(lambda r, v: cuts, bucketer=per_tier_bucket)
        return [
            (r.round_index, r.tier_index, r.flag.value, r.count)
            for r in out.transcript.view().all()
        ]

    assert run() == run()


def test_higher_m_min_suppresses_more():
    lcfg = LatentConfig()

    def releases(m):
        eng = Engine(
            EngineConfig(seed=42, num_devices=1000, num_rounds=30,
                         round_config=RoundConfig(m_min=m)),
            lcfg,
        )
        cuts = eng.calibrate_fixed_deadlines((0.33, 0.66))
        out = eng.run(lambda r, v: cuts, bucketer=per_tier_bucket)
        return len(out.transcript.view().released())

    assert releases(200) < releases(16)
