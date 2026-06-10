"""Shared experiment defaults — single source of truth.

Every experiment imports these so that device count, tier count K, dataset, and
seed are IDENTICAL across the whole evaluation, unless an experiment is directly
studying one of them (e.g. the m_min sweep varies m_min; the headline frontier
varies m_min and K on purpose). When an experiment overrides a value, it should
do so explicitly and locally, not by hardcoding a different default.

Rationale for the chosen values:
  - DEVICES = 1500: large enough that capability-inference advantage and
    anonymity-set statistics are stable, yet tractable for real local-SGD
    training runs. (Privacy experiments previously used 3000, training used 300;
    1500 is the common ground that keeps both meaningful.)
  - TIERS (K) = 5: the tier count used throughout the privacy results; the
    headline frontier additionally sweeps K to show it is immaterial vs
    participation, but every other experiment fixes K = 5.
  - DATASET = "synthetic": default learning task. Real-data runs pass
    --data cifar10 / cifar100 explicitly; that is a deliberate override, not a
    silent inconsistency.
  - SEED = 42: base RNG seed. Multi-seed experiments derive a fixed list from
    this base (SEEDS) rather than inventing their own.
  - ROUNDS is intentionally NOT globalized to a single value: privacy attacks
    need only enough rounds to accumulate a transcript (PRIVACY_ROUNDS), while
    learning-curve experiments need many more to converge (TRAINING_ROUNDS).
    Both are derived here so they are still centrally controlled.
"""
from __future__ import annotations

# --- core shared parameters (the same everywhere unless directly studied) ---
DEVICES: int = 1500
TIERS: int = 5
DATASET: str = "synthetic"
SEED: int = 42

# --- multi-seed list, derived from the base seed ---
NUM_SEEDS: int = 5
SEEDS: tuple[int, ...] = tuple(SEED + i for i in range(NUM_SEEDS))

# --- round budgets (centrally controlled, but task-dependent) ---
PRIVACY_ROUNDS: int = 80      # enough to accumulate a transcript for attacks
TRAINING_ROUNDS: int = 100    # enough for learning curves to converge

# --- learning defaults shared by training / utility / baseline experiments ---
HIDDEN: int = 64
LR: float = 0.05
TARGET_ACC: float = 0.5       # default target for time-to-accuracy metrics
