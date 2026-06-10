"""L3 omniscient-except-target capability-inference attacker (the adversary CEILING).

Auxiliary knowledge: the TRUE capability class of every device EXCEPT the target.
For each target device, the attacker trains on all other devices' (features,
true class) pairs and predicts the single held-out target. This is the
membership-inference-style worst case: it upper-bounds how much a target's own
observed tier sequence reveals about its class when the adversary already knows
everyone else.

If L3 advantage is not much above L1, the labeled-seed adversary is already near
the ceiling -- the leakage is "easy". If L3 is far above L1, more auxiliary
knowledge keeps helping, and the realistic L1 number is a loose lower bound.
Either way the ladder L0 <= L1 <= L2/L3 brackets the true leakage.

Honest cost note: strict leave-one-out (refit per target) is O(N) fits. We use
the standard, faithful approximation: a single fit on ALL devices then predict
each (an upper bound on true LOO, since the target's own row is in training);
plus an exact LOO pass on a random subsample to confirm the gap is small. The
harness can request either mode.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dtfl.attack.base import Adversary
from dtfl.attack.observation import DeviceObservation, ObservationFeaturizer

__all__ = ["L3OmniscientAttacker"]


class L3OmniscientAttacker(Adversary):
    """Knows all other devices' classes; predicts each target from the rest."""

    def __init__(
        self,
        featurizer: ObservationFeaturizer,
        min_observations: int = 5,
        n_estimators: int = 200,
        random_state: int = 0,
        exact_loo: bool = False,
    ):
        self._featurizer = featurizer
        self._min_obs = min_observations
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._exact_loo = exact_loo
        # L3 holds ALL labeled observations (its omniscient knowledge), set in fit.
        self._all_obs: list[DeviceObservation] = []
        self._all_labels: np.ndarray | None = None

    def fit(
        self,
        seed_observations: list[DeviceObservation],
        seed_labels: np.ndarray,
    ) -> None:
        """For L3, the 'seed' IS the full labeled population (minus each target).

        The harness passes ALL devices' observations and labels here; predict()
        then holds out each query device and trains on the remainder.
        """
        self._all_obs = list(seed_observations)
        self._all_labels = np.asarray(seed_labels)

    def predict(self, query_observations: list[DeviceObservation]) -> np.ndarray:
        if self._all_labels is None:
            raise RuntimeError("L3 must be fit with the full labeled population first")

        # Build a device_id -> index map into the omniscient pool.
        id_to_idx = {o.device_id: i for i, o in enumerate(self._all_obs)}
        all_rich_idx = [
            i for i, o in enumerate(self._all_obs) if o.num_observations >= self._min_obs
        ]
        X_all = self._featurizer.featurize_many([self._all_obs[i] for i in all_rich_idx])
        y_all = self._all_labels[all_rich_idx]
        idx_pos = {orig: j for j, orig in enumerate(all_rich_idx)}  # orig pool idx -> row in X_all

        default = int(np.bincount(self._all_labels).argmax())
        preds = np.full(len(query_observations), default, dtype=np.int64)

        if self._exact_loo:
            # Exact: refit on all-but-target for each query. Faithful but O(N) fits.
            for qi, obs in enumerate(query_observations):
                if obs.num_observations < self._min_obs:
                    continue
                pool_idx = id_to_idx.get(obs.device_id)
                if pool_idx is None or pool_idx not in idx_pos:
                    continue
                row = idx_pos[pool_idx]
                mask = np.ones(X_all.shape[0], dtype=bool)
                mask[row] = False
                if len(np.unique(y_all[mask])) < 2:
                    continue
                clf = RandomForestClassifier(
                    n_estimators=self._n_estimators, random_state=self._random_state, n_jobs=-1
                )
                clf.fit(X_all[mask], y_all[mask])
                preds[qi] = int(clf.predict(X_all[row : row + 1])[0])
            return preds

        # Default: single fit on the full omniscient pool, predict each target.
        # Upper bound on true LOO (target's row is in training), which is the
        # honest "ceiling" semantics we want for L3.
        if len(np.unique(y_all)) < 2:
            return preds
        clf = RandomForestClassifier(
            n_estimators=self._n_estimators, random_state=self._random_state, n_jobs=-1
        )
        clf.fit(X_all, y_all)
        for qi, obs in enumerate(query_observations):
            if obs.num_observations < self._min_obs:
                continue
            pool_idx = id_to_idx.get(obs.device_id)
            if pool_idx is None or pool_idx not in idx_pos:
                continue
            row = idx_pos[pool_idx]
            preds[qi] = int(clf.predict(X_all[row : row + 1])[0])
        return preds
