"""Multi-seed replication and confidence-interval aggregation.

Every result so far is single-seed. A reviewer (rightly) wants to know whether
the numbers are stable across a different random deployment, not just a different
data split. This module re-runs any single-point evaluation across S seeds and
aggregates into mean / std / 95% CI.

What "seed" varies:
  - the ENGINE seed (population, availability, per-round latency/dropout draws),
  - the ATTACK seed (train/query split, classifier RNG).
We vary BOTH together per replicate (replicate r uses engine_seed and
attack_seed both derived from a base via the project's stable substream hashing),
so each replicate is an independent random deployment + independent attack on it.
That captures the full variance a reader cares about.

The driver is generic: pass a callable that takes (engine_seed, attack_seed) and
returns a dict of scalar metrics; get back per-metric aggregates across seeds.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from dtfl.rng import stable_substream_seed

__all__ = ["MetricAgg", "MultiSeedResult", "run_multiseed", "seed_pairs"]


@dataclass
class MetricAgg:
    """Aggregate of one scalar metric across replicates."""

    name: str
    values: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else 0.0

    @property
    def sem(self) -> float:
        return self.std / np.sqrt(len(self.values)) if len(self.values) > 1 else 0.0

    def ci95(self) -> tuple[float, float]:
        """95% confidence interval for the mean (t-distribution, small S)."""
        n = len(self.values)
        if n < 2:
            return (self.mean, self.mean)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        half = tcrit * self.sem
        return (self.mean - half, self.mean + half)

    @property
    def ci_halfwidth(self) -> float:
        lo, hi = self.ci95()
        return (hi - lo) / 2.0

    def summary(self) -> str:
        lo, hi = self.ci95()
        return f"{self.name}: {self.mean:.4f} +/- {self.ci_halfwidth:.4f}  (95% CI [{lo:.4f}, {hi:.4f}], n={len(self.values)})"


@dataclass
class MultiSeedResult:
    """All metrics aggregated across replicates, keyed by metric name."""

    aggregates: dict[str, MetricAgg] = field(default_factory=dict)
    raw: list[dict[str, float]] = field(default_factory=list)  # per-replicate dicts

    def add_replicate(self, metrics: dict[str, float]) -> None:
        self.raw.append(metrics)

    def finalize(self) -> MultiSeedResult:
        if not self.raw:
            return self
        names = self.raw[0].keys()
        for name in names:
            vals = [r[name] for r in self.raw if name in r]
            self.aggregates[name] = MetricAgg(name=name, values=vals)
        return self

    def summary(self) -> str:
        return "\n".join(self.aggregates[n].summary() for n in self.aggregates)


def seed_pairs(base_seed: int, num_seeds: int) -> list[tuple[int, int]]:
    """Generate (engine_seed, attack_seed) pairs for ``num_seeds`` replicates.

    Both derived via the project's stable name-hashing so the set of seeds is
    deterministic given the base, and adding more replicates never changes the
    earlier ones (reproducibility: a paper rerun with S=5 reproduces the first 3
    of an S=3 run exactly).
    """
    pairs = []
    for i in range(num_seeds):
        eng = stable_substream_seed(base_seed, f"engine.replicate.{i}")
        atk = stable_substream_seed(base_seed, f"attack.replicate.{i}")
        pairs.append((eng, atk))
    return pairs


def run_multiseed(
    eval_fn: Callable[[int, int], dict[str, float]],
    num_seeds: int = 5,
    base_seed: int = 2024,
    verbose: bool = True,
) -> MultiSeedResult:
    """Run ``eval_fn(engine_seed, attack_seed)`` across replicates and aggregate.

    ``eval_fn`` returns a flat dict of scalar metrics; all are aggregated.
    """
    result = MultiSeedResult()
    for i, (eng_seed, atk_seed) in enumerate(seed_pairs(base_seed, num_seeds)):
        metrics = eval_fn(eng_seed, atk_seed)
        result.add_replicate(metrics)
        if verbose:
            shown = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            print(f"  replicate {i+1}/{num_seeds} (eng={eng_seed % 100000}): {shown}")
    return result.finalize()
