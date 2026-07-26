"""Tests for the FeDSC clustering baseline (dtfl.controller.fedsc).

These pin the properties Algorithm 1 of Liu et al. (T-COMM 2024) specifies, so a
future refactor cannot silently change what the baseline does.
"""
from __future__ import annotations

import numpy as np

from dtfl.controller.fedsc import FeDSCClusterer, FeDSCController, _CF
from dtfl.protocol.tiering import assign_tiers
from dtfl.types import RoundDeadlines


def test_cf_delta_matches_direct_formula():
    """T^Delta from the CF triple must equal the direct O(n^2) definition."""
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 7)
    cf = _CF(x[0])
    for v in x[1:-1]:
        cf.add(v)
    got = cf.delta_if_added(x[-1])
    n = x.size
    direct = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum() / (n * (n - 1)))
    assert np.isclose(got, direct)


def test_separated_groups_recovered():
    """Well-separated time groups should come back as distinct clusters."""
    rng = np.random.default_rng(1)
    t = np.concatenate([
        rng.normal(1.0, 0.02, 40),
        rng.normal(5.0, 0.02, 40),
        rng.normal(9.0, 0.02, 40),
    ])
    cents = FeDSCClusterer(t_bar=0.5, beta=10**9).fit(t)
    assert len(cents) == 3
    assert np.allclose(sorted(cents), [1.0, 5.0, 9.0], atol=0.05)


def test_t_bar_monotonicity():
    """Larger t_bar admits looser clusters, so the count is non-increasing."""
    rng = np.random.default_rng(2)
    t = rng.normal(5.0, 1.0, 300)
    counts = [len(FeDSCClusterer(tb, 10**9).fit(t)) for tb in (0.2, 0.5, 1.0, 2.0, 5.0)]
    assert counts == sorted(counts, reverse=True)


def test_beta_bounds_cluster_size():
    """No cluster may exceed the branching factor beta."""
    rng = np.random.default_rng(3)
    t = rng.normal(5.0, 1.0, 200)
    pairs = FeDSCClusterer(t_bar=10**6, beta=10).fit_sizes(t)
    assert max(n for _, n in pairs) <= 10


def test_controller_emits_valid_cutoffs():
    """Cutoffs must be strictly increasing and of length K-1 (engine contract)."""
    rng = np.random.default_rng(4)
    t = np.exp(rng.normal(5.0, 0.35, 400))
    K = 5
    ctl = FeDSCController(num_tiers=K, recluster_every=5)
    cuts = ctl.next_deadlines_from_times(0, t)
    assert len(cuts) == K - 1
    assert all(cuts[i] < cuts[i + 1] for i in range(len(cuts) - 1))
    # partition must be usable by the engine's tiering. -1 is the engine's
    # MISSED sentinel (device slower than the last deadline), so valid labels
    # are {-1} U {0..K-1}.
    tiers = assign_tiers(t, RoundDeadlines(round_index=0, cutoffs=tuple(cuts)))
    assert tiers.min() >= -1 and tiers.max() < K
    assert (tiers >= 0).any()  # at least someone is placed in a real tier


def test_controller_reclusters_on_schedule():
    """Cutoffs are cached between re-clustering rounds (the paper's constant c)."""
    rng = np.random.default_rng(5)
    t1 = np.exp(rng.normal(5.0, 0.35, 300))
    t2 = np.exp(rng.normal(6.0, 0.35, 300))  # shifted population
    ctl = FeDSCController(num_tiers=5, recluster_every=5)
    c0 = ctl.next_deadlines_from_times(0, t1)
    c1 = ctl.next_deadlines_from_times(1, t2)   # no recluster -> cached
    assert c0 == c1
    c5 = ctl.next_deadlines_from_times(5, t2)   # recluster round
    assert c5 != c0


def test_controller_handles_empty_round():
    """An empty availability draw must not crash the controller."""
    ctl = FeDSCController(num_tiers=4, recluster_every=1)
    cuts = ctl.next_deadlines_from_times(0, np.array([]))
    assert len(cuts) == 3
    assert all(cuts[i] < cuts[i + 1] for i in range(len(cuts) - 1))