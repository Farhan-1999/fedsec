"""Latent layer tests: population, latency draw, the SNR knob, drift, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from dtfl.latent import (
    DriftConfig,
    DriftState,
    LatentConfig,
    build_population,
    class_tier_mutual_information,
    decompose,
    draw_completion_times,
)
from dtfl.rng import RngHub


@pytest.fixture
def cfg():
    return LatentConfig()


@pytest.fixture
def population(cfg):
    hub = RngHub(seed=12345)
    pop = build_population(1000, cfg, hub.stream("latent.population"))
    mu = np.array([d.mu for d in pop])
    classes = np.array([d.capability_class for d in pop])
    return pop, mu, classes


def test_population_size_and_class_range(population, cfg):
    pop, _, classes = population
    assert len(pop) == 1000
    assert classes.min() >= 1 and classes.max() <= cfg.num_classes


def test_class_means_monotone_increasing(population):
    _, mu, classes = population
    means = [mu[classes == c].mean() for c in range(1, classes.max() + 1)]
    assert all(np.diff(means) > 0), means


def test_latency_positive_and_finite(population, cfg):
    _, mu, _ = population
    tau = draw_completion_times(mu, cfg, RngHub(seed=1).stream("lat"))
    assert np.all(tau > 0) and np.all(np.isfinite(tau))


def test_median_latency_increases_with_class(population, cfg):
    _, mu, classes = population
    tau = draw_completion_times(mu, cfg, RngHub(seed=1).stream("lat"))
    meds = [np.median(tau[classes == c]) for c in range(1, classes.max() + 1)]
    assert all(np.diff(meds) > 0), meds


def test_decomposition_sums_to_total(population, cfg):
    _, mu, _ = population
    tau = draw_completion_times(mu, cfg, RngHub(seed=1).stream("lat"))
    assert np.allclose(decompose(tau).total, tau)


def test_snr_knob_lowers_mutual_information(population, cfg):
    """Raising proxy noise eta must monotonically reduce class->tier MI."""
    _, mu, classes = population
    base_tau = draw_completion_times(mu, cfg.with_eta(0.05), RngHub(seed=2).stream("b"))
    deadlines = np.append(np.quantile(base_tau, [0.33, 0.66]), base_tau.max() * 1.5)

    def assign(tau):
        t = np.full(tau.shape[0], -1, dtype=np.int64)
        for k, d in enumerate(deadlines):
            t[(t == -1) & (tau <= d)] = k
        return t

    mis = []
    for e in [0.05, 0.15, 0.30, 0.50, 0.80]:
        tau_e = draw_completion_times(mu, cfg.with_eta(e), RngHub(seed=999).stream(f"e{e}"))
        mis.append(class_tier_mutual_information(classes, assign(tau_e)))
    # non-increasing (small tolerance) and a meaningful overall drop
    assert all(mis[i] >= mis[i + 1] - 0.01 for i in range(len(mis) - 1)), mis
    assert mis[0] > mis[-1] + 0.05, mis


def test_determinism_same_seed_same_tau(population, cfg):
    _, mu, _ = population
    a = draw_completion_times(mu, cfg, RngHub(seed=7).stream("x"))
    b = draw_completion_times(mu, cfg, RngHub(seed=7).stream("x"))
    assert np.array_equal(a, b)


def test_drift_disabled_is_noop(population):
    _, mu, _ = population
    d = DriftState(DriftConfig(), n_devices=mu.shape[0], rng=RngHub(seed=1).stream("d"))
    assert d.step() == 0.0
    assert np.array_equal(d.apply_device_walk(mu), mu)


def test_drift_enabled_moves_state(population):
    _, mu, _ = population
    d = DriftState(
        DriftConfig(slow_drift_per_round=0.05, device_walk_per_round=0.02),
        n_devices=mu.shape[0],
        rng=RngHub(seed=1).stream("d2"),
    )
    assert d.step() != 0.0
    assert not np.array_equal(d.apply_device_walk(mu), mu)
