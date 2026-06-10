"""L2 statistical-prior capability-inference attacker.

Auxiliary knowledge: the TRUE generative latency model -- per-class mean
log-latency, proxy-noise scale, the population class mixture, and the round
deadlines -- but NO per-device labels. This models an adversary who knows how the
system works statistically (e.g. from public documentation or offline profiling
of representative devices) but cannot label specific live devices.

Method: Bayes-optimal posterior over classes given a device's observed tier
counts. For a device of class c, the per-round probability of landing in tier k
is the probability its completion time falls in (d_{k-1}, d_k], computed from the
log-normal model log tau ~ Normal(m_c, sqrt(s_within^2 + eta^2)). Assuming rounds
are conditionally independent given the class, the likelihood of an observed
tier-count vector is multinomial, and the posterior combines it with the class
mixture prior.

This is an UPPER-MIDDLE rung: it tests how much leaks when the adversary has
perfect model knowledge but no labels. It uses only model parameters that are
passed in explicitly -- it does not read any device's latent state, preserving
the separation boundary (model knowledge != per-device ground truth).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from dtfl.attack.base import Adversary
from dtfl.attack.observation import DeviceObservation

__all__ = ["L2PriorAttacker", "ModelKnowledge"]


class ModelKnowledge:
    """The generative-model parameters the L2 adversary is assumed to know.

    Constructed by the harness from the (public-by-assumption) LatentConfig and
    the round deadlines. Passing these explicitly -- rather than handing the
    attacker a LatentConfig object -- keeps the attack package free of any latent
    import and makes the assumed knowledge auditable.
    """

    def __init__(
        self,
        class_means_log: np.ndarray,  # m_c for c=1..C, shape (C,)
        latency_sigma_log: float,  # sqrt(s_within^2 + eta^2)
        class_mixture: np.ndarray,  # prior over classes, shape (C,)
        deadlines: np.ndarray,  # tier cutoffs, shape (K,), strictly increasing
    ):
        self.class_means_log = np.asarray(class_means_log, dtype=np.float64)
        self.latency_sigma_log = float(latency_sigma_log)
        self.class_mixture = np.asarray(class_mixture, dtype=np.float64)
        self.deadlines = np.asarray(deadlines, dtype=np.float64)
        self.C = self.class_means_log.size
        self.K = self.deadlines.size

    def tier_probability_matrix(self) -> np.ndarray:
        """P(tier k | class c) for all c, k. Shape (C, K).

        P(land in tier k | class c) = P(d_{k-1} < tau <= d_k) under
        log tau ~ Normal(m_c, sigma). Tier 0 is (0, d_0]; tier k is (d_{k-1}, d_k].
        Mass beyond the last deadline is a missed-deadline event (not a tier), so
        rows need not sum to 1; we renormalize over observed tiers at scoring.
        """
        P = np.zeros((self.C, self.K), dtype=np.float64)
        log_cuts = np.log(self.deadlines)
        for c in range(self.C):
            cdf = norm.cdf(log_cuts, loc=self.class_means_log[c], scale=self.latency_sigma_log)
            # tier 0: cdf at d_0; tier k: cdf(d_k) - cdf(d_{k-1})
            P[c, 0] = cdf[0]
            if self.K > 1:
                P[c, 1:] = np.diff(cdf)
        # Guard against zeros (log later): floor tiny probabilities.
        return np.clip(P, 1e-12, None)


class L2PriorAttacker(Adversary):
    """Bayes-optimal class posterior from known model; no per-device labels."""

    def __init__(self, knowledge: ModelKnowledge, num_tiers: int, min_observations: int = 5):
        self._k = knowledge
        self._num_tiers = num_tiers
        self._min_obs = min_observations
        self._log_tier_prob = np.log(knowledge.tier_probability_matrix())  # (C, K)
        self._log_prior = np.log(np.clip(knowledge.class_mixture, 1e-12, None))  # (C,)

    def fit(
        self,
        seed_observations: list[DeviceObservation] | None = None,
        seed_labels: np.ndarray | None = None,
    ) -> None:
        """No-op: L2 is fully specified by its model knowledge; it does not train."""
        return

    def _tier_counts(self, obs: DeviceObservation) -> np.ndarray:
        counts = np.zeros(self._num_tiers, dtype=np.float64)
        for _, k in obs.tier_sequence:
            if 0 <= k < self._num_tiers:
                counts[k] += 1.0
        return counts

    def predict(self, query_observations: list[DeviceObservation]) -> np.ndarray:
        n = len(query_observations)
        # Default = mixture mode (most likely class a priori).
        default = int(np.argmax(self._k.class_mixture)) + 1
        preds = np.full(n, default, dtype=np.int64)
        for i, obs in enumerate(query_observations):
            if obs.num_observations < self._min_obs:
                continue
            counts = self._tier_counts(obs)  # (K,)
            # log posterior_c proportional to log_prior_c + sum_k counts_k * log P(k|c)
            log_post = self._log_prior + self._log_tier_prob @ counts  # (C,)
            preds[i] = int(np.argmax(log_post)) + 1
        return preds
