"""L0 unsupervised capability-inference attacker (the adversary FLOOR).

Auxiliary knowledge: NONE. No labeled seed, no generative model. The attacker
sees only the observable tier-sequence features (same featurizer as L1) and must
recover capability structure with zero supervision.

Method: cluster device feature vectors into C groups (KMeans), then label
clusters by their mean occupied-tier index -- the cluster whose devices occupy
the fastest (lowest-index) tiers is mapped to the fastest class, and so on. This
ordering assumption is the only "knowledge" L0 uses, and it is generic (faster
tiers <-> higher capability is the system's defining premise, not a secret).

L0 is the floor: if even an unsupervised attacker recovers capability, the
leakage is intrinsic to the released structure, not an artifact of the seed. If
L0 fails but L1 succeeds, the labeled seed is doing the work -- a meaningful
distinction the ladder makes visible.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from dtfl.attack.base import Adversary
from dtfl.attack.observation import DeviceObservation, ObservationFeaturizer

__all__ = ["L0UnsupervisedAttacker"]


class L0UnsupervisedAttacker(Adversary):
    """KMeans clustering + tier-order cluster labeling. No labels used."""

    def __init__(
        self,
        featurizer: ObservationFeaturizer,
        num_classes: int,
        min_observations: int = 5,
        random_state: int = 0,
    ):
        self._featurizer = featurizer
        self._C = num_classes
        self._min_obs = min_observations
        self._random_state = random_state
        self._kmeans: KMeans | None = None
        self._cluster_to_class: dict[int, int] = {}
        # occupancy feature indices are the first C entries (see featurizer)
        self._num_tiers = featurizer.num_tiers

    def fit(
        self,
        seed_observations: list[DeviceObservation],
        seed_labels: np.ndarray | None = None,  # ignored: L0 uses no labels
    ) -> None:
        """Cluster the (seed) observations. Labels are accepted but NOT used.

        For L0 the "seed" is just the pool of observations to cluster on; in
        practice we cluster on ALL observed devices at predict time, so fit may
        receive the full observation set. seed_labels is ignored by design.
        """
        rich = [o for o in seed_observations if o.num_observations >= self._min_obs]
        if len(rich) < self._C:
            self._kmeans = None
            return
        X = self._featurizer.featurize_many(rich)
        self._kmeans = KMeans(n_clusters=self._C, n_init=10, random_state=self._random_state)
        cluster_ids = self._kmeans.fit_predict(X)

        # Label clusters by mean occupied-tier index (fastest tier -> class 1).
        # Occupancy histogram occupies feature indices [0, num_tiers).
        occ = X[:, : self._num_tiers]
        tier_index = np.arange(self._num_tiers)
        # expected tier per device = sum_k k * occupancy_k
        exp_tier = occ @ tier_index
        cluster_mean_tier = {}
        for c in range(self._C):
            mask = cluster_ids == c
            cluster_mean_tier[c] = float(exp_tier[mask].mean()) if mask.any() else np.inf
        # Sort clusters by mean tier; lowest mean tier -> class 1 (fastest).
        order = sorted(cluster_mean_tier, key=lambda c: cluster_mean_tier[c])
        self._cluster_to_class = {c: rank + 1 for rank, c in enumerate(order)}

    def predict(self, query_observations: list[DeviceObservation]) -> np.ndarray:
        n = len(query_observations)
        # Default prediction: middle class (no information).
        default = (self._C + 1) // 2
        preds = np.full(n, default, dtype=np.int64)
        if self._kmeans is None:
            return preds
        rich_idx = [
            i for i, o in enumerate(query_observations) if o.num_observations >= self._min_obs
        ]
        if not rich_idx:
            return preds
        X = self._featurizer.featurize_many([query_observations[i] for i in rich_idx])
        clusters = self._kmeans.predict(X)
        for j, i in enumerate(rich_idx):
            preds[i] = self._cluster_to_class.get(int(clusters[j]), default)
        return preds
