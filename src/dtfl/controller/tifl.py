"""TiFL-style adaptive tier selection (Chai et al., SC 2020) — utility baseline.

TiFL is the closest prior work: it groups clients into tiers by measured response
latency, then each round SELECTS ONE TIER to train (not all tiers merged), using
an adaptive, credit-based scheme that favors faster tiers for speed but
periodically samples slower tiers so they are not starved (which would bias the
model). Crucially, TiFL has NO privacy mechanism — no m_min suppression, no
anonymity floor; it uses individual clients freely.

This module provides a tier-selection callable compatible with
federated_train(tier_selector=...). It is a BASELINE for the utility comparison,
not part of our framework, and it deliberately lives in controller/ (it sets who
trains) — it never reads latent state, only the round index and a credit table it
maintains from observed participation, so it stays transcript-compatible.

Faithful-but-simple selection: each tier has a weight; the round's tier is drawn
with probability proportional to weight. Faster tiers (lower index) get higher
base weight (they finish sooner -> cheaper rounds), but every tier keeps a floor
probability so none is starved. This captures TiFL's speed/coverage trade-off
without reproducing their exact credit-update math, which depends on per-tier
test accuracy they measure online.
"""
from __future__ import annotations

import numpy as np

__all__ = ["TiFLSelector"]


class TiFLSelector:
    """Adaptive single-tier-per-round selector in the style of TiFL.

    Parameters
    ----------
    num_tiers:
        Number of latency tiers.
    speed_bias:
        How strongly to favor faster (lower-index) tiers. 0 = uniform over tiers;
        higher = stronger preference for fast tiers. TiFL favors fast tiers for
        wall-clock efficiency.
    floor:
        Minimum selection probability mass reserved equally across all tiers so no
        tier is ever starved (TiFL's anti-starvation guarantee).
    """

    def __init__(self, num_tiers: int, speed_bias: float = 1.0, floor: float = 0.10):
        self.num_tiers = num_tiers
        # base weights decay with tier index (tier 0 fastest). Geometric decay.
        idx = np.arange(num_tiers, dtype=np.float64)
        base = np.exp(-speed_bias * idx / max(1, num_tiers - 1))
        base = base / base.sum()
        # mix with a uniform floor so slow tiers retain coverage
        unif = np.full(num_tiers, 1.0 / num_tiers)
        self._probs = (1.0 - floor) * base + floor * unif
        self._probs = self._probs / self._probs.sum()

    def __call__(self, round_index: int, num_tiers: int, rng: np.random.Generator):
        """Return a one-element list: the tier index to train this round."""
        k = int(rng.choice(num_tiers, p=self._probs[:num_tiers] / self._probs[:num_tiers].sum()))
        return [k]


class AdaptiveTiFLSelector:
    """Full adaptive TiFL selection with credits + accuracy-driven probabilities.

    Faithful to the adaptive variant of Chai et al. (SC 2020):

      - Training proceeds in INTERVALS of ``interval`` rounds. At the start of each
        interval every tier is given a CREDIT budget (how many rounds it may be
        chosen this interval); a tier with zero remaining credit is skipped. This
        bounds how often any single tier is trained per interval (anti-monopoly).

      - Each tier carries a selection WEIGHT. Within an interval the next tier is
        drawn among credit-bearing tiers with probability proportional to weight.

      - At each interval boundary the weights are RE-COMPUTED from the tiers'
        recently observed accuracies: tiers whose accuracy is LAGGING get higher
        weight, so the scheme spends more rounds on under-trained (typically
        slower / less-sampled) tiers and avoids overfitting the model to the fast
        tiers. This is TiFL's accuracy-aware adaptation, the part the simple
        speed-biased selector omits.

    The loop reports accuracy back via ``update(selected, accuracy, round)`` after
    every round; the selector attributes that accuracy to the tier it picked and
    uses the per-tier accuracy history to re-weight at interval boundaries.

    Reads no latent state — only the round index, the credits it maintains, and
    the scalar validation accuracy the loop already computes. It is a utility
    BASELINE, not part of the privacy framework.
    """

    def __init__(
        self,
        num_tiers: int,
        interval: int = 5,
        base_credit: int | None = None,
        seed: int = 0,
    ):
        self.num_tiers = num_tiers
        self.interval = max(1, interval)
        # default per-tier credit so the interval can be filled across tiers
        self.base_credit = base_credit if base_credit is not None else max(1, interval)
        self._rng = np.random.default_rng(seed)
        self._weights = np.full(num_tiers, 1.0 / num_tiers)
        self._credits = np.full(num_tiers, self.base_credit, dtype=int)
        # last observed accuracy attributed to each tier (NaN = not yet seen)
        self._tier_acc = np.full(num_tiers, np.nan)
        self._last_pick: int | None = None
        self._interval_start = 0

    def _reset_interval(self):
        self._credits[:] = self.base_credit
        # Re-weight from observed accuracies: lower accuracy -> higher weight.
        acc = self._tier_acc.copy()
        if np.all(np.isnan(acc)):
            self._weights[:] = 1.0 / self.num_tiers
            return
        # fill unseen tiers with the mean so they get average priority
        mean_acc = np.nanmean(acc)
        acc = np.where(np.isnan(acc), mean_acc, acc)
        # weight inversely to accuracy gap above the worst tier; add a floor so
        # well-performing tiers still get sampled occasionally.
        deficit = (acc.max() - acc) + 1e-3
        w = deficit / deficit.sum()
        floor = 0.10
        self._weights = (1.0 - floor) * w + floor * (1.0 / self.num_tiers)
        self._weights /= self._weights.sum()

    def __call__(self, round_index: int, num_tiers: int, rng: np.random.Generator):
        if round_index - self._interval_start >= self.interval:
            self._interval_start = round_index
            self._reset_interval()
        eligible = np.flatnonzero(self._credits[:num_tiers] > 0)
        if eligible.size == 0:
            self._credits[:] = self.base_credit
            eligible = np.arange(num_tiers)
        w = self._weights[eligible]
        w = w / w.sum()
        k = int(self._rng.choice(eligible, p=w))
        self._credits[k] -= 1
        self._last_pick = k
        return [k]

    def update(self, selected, accuracy: float, round_index: int):
        """Attribute the post-round accuracy to the tier just trained."""
        if self._last_pick is not None:
            # EWMA so a tier's estimate reflects recent rounds, not just the last
            prev = self._tier_acc[self._last_pick]
            self._tier_acc[self._last_pick] = (
                accuracy if np.isnan(prev) else 0.5 * prev + 0.5 * accuracy
            )
