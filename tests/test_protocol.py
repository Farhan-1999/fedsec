"""Protocol layer tests: the two conflict-resolution invariants are the core.

  - t <= m_min coupling rule (preamble Section 6),
  - size-weighted merge == FedAvg-on-participants (preamble Section 7).
"""

from __future__ import annotations

import numpy as np
import pytest

from dtfl.latent import LatentConfig, build_population, draw_completion_times
from dtfl.protocol import (
    DropoutRates,
    TierContribution,
    apply_dropout,
    apply_server_update,
    assign_tiers,
    decide_release,
    emit_record,
    reconstruction_threshold,
    safe_coupling_holds,
    size_weighted_merge,
    tier_rosters,
    weighted_merge,
)
from dtfl.protocol.tiering import MISSED
from dtfl.rng import RngHub
from dtfl.types import RoundDeadlines


@pytest.fixture
def setup():
    hub = RngHub(seed=2024)
    cfg = LatentConfig()
    pop = build_population(2000, cfg, hub.stream("pop"))
    mu = np.array([d.mu for d in pop])
    classes = np.array([d.capability_class for d in pop])
    tau = draw_completion_times(mu, cfg, hub.stream("lat"))
    cutoffs = tuple(np.append(np.quantile(tau, [0.33, 0.66]), tau.max() * 1.5))
    deadlines = RoundDeadlines(0, cutoffs)
    return hub, classes, tau, deadlines


def test_tiering_assigns_faster_to_earlier_tiers(setup):
    _, classes, tau, deadlines = setup
    tiers = assign_tiers(tau, deadlines)
    assert (tiers == MISSED).sum() == 0  # covering last tier
    assert classes[tiers == 0].mean() < classes[tiers == 2].mean()


def test_rosters_partition_participants(setup):
    _, _, tau, deadlines = setup
    tiers = assign_tiers(tau, deadlines)
    rosters = tier_rosters(tiers, 3)
    assert sum(len(r) for r in rosters) == (tiers != MISSED).sum()


def test_dropout_orders_roster_active_unmask(setup):
    hub, _, tau, deadlines = setup
    rates = DropoutRates(rho_mask=0.10, rho_unmask=0.05)
    oc = apply_dropout(500, rates, hub.stream("drop"))
    assert oc.roster_size >= oc.active_count >= oc.unmask_count


@pytest.mark.parametrize("n_k", [1, 5, 50, 500, 2000])
@pytest.mark.parametrize("m_min", [8, 16, 32, 64])
def test_threshold_coupling_invariant(n_k, m_min):
    """t <= m_min for every tier size -- the safe-coupling invariant."""
    rates = DropoutRates()
    t = reconstruction_threshold(n_k, m_min, rates)
    assert safe_coupling_holds(t, m_min), (n_k, m_min, t)


def test_release_suppresses_below_m_min(setup):
    hub, _, _, deadlines = setup
    rates = DropoutRates()
    small = apply_dropout(5, rates, hub.stream("s"))
    t = reconstruction_threshold(small.active_count, 16, rates)
    dec = decide_release(small, 16, t)
    assert not dec.released and dec.reason == "below_m_min"


def test_suppressed_record_hides_count_and_sum(setup):
    hub, _, _, deadlines = setup
    rates = DropoutRates()
    small = apply_dropout(5, rates, hub.stream("s"))
    t = reconstruction_threshold(small.active_count, 16, rates)
    dec = decide_release(small, 16, t)
    rec = emit_record(0, 0, dec, deadlines, secure_sum=np.ones(4), release_bucket=0)
    assert rec.count is None and rec.secure_sum is None


def test_fedavg_equivalence(setup):
    """Size-weighted merge of tier sums == plain mean over the union of clients."""
    hub, _, tau, deadlines = setup
    tiers = assign_tiers(tau, deadlines)
    rosters = tier_rosters(tiers, 3)
    rates = DropoutRates()
    updates = hub.stream("u").normal(size=(2000, 8))
    contribs, union = [], []
    for k in range(3):
        oc = apply_dropout(len(rosters[k]), rates, hub.stream(f"d{k}"))
        active = rosters[k][oc.active]
        if active.size == 0:
            continue
        contribs.append(TierContribution(updates[active].sum(axis=0), active.size))
        union.append(active)
    merged = size_weighted_merge(contribs)
    fedavg = updates[np.concatenate(union)].mean(axis=0)
    assert np.allclose(merged, fedavg, atol=1e-10)


def test_weighted_merge_reduces_to_size_weighted(setup):
    c = [TierContribution(np.ones(4) * 3, 3), TierContribution(np.ones(4) * 5, 5)]
    assert np.allclose(weighted_merge(c, 0.0), size_weighted_merge(c), atol=1e-10)


def test_no_participation_update_is_noop():
    w = np.ones(4)
    w2, _ = apply_server_update(w, None, lr=0.5)
    assert np.array_equal(w2, w)
