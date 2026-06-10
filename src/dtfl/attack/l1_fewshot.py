"""L1 few-shot capability-inference attacker (the primary adversary).

Threat model (adversary spec, rung L1): the attacker controls/observes a small
fraction of devices whose true capability class it knows -- the labeled seed. For
every device it observes (seed and target alike) it sees the device's per-round
TIER SEQUENCE, which is observable to an adversary co-located with that device.
The hidden target is the device's CAPABILITY CLASS.

The attack: featurize each observed device's tier sequence (occupancy, summary
stats, transitions, and transcript bucket statistics -- all observable), train a
classifier on the labeled seed, predict classes for the rest. If the observable
tier sequence carries the hidden class, advantage > 0 and capability leaks.

This is honest because:
  - it uses only observable inputs (tier sequences + transcript), never the
    latent class at inference,
  - labels are supplied only for the seed (the assumed auxiliary knowledge),
  - the same defenses that blur tier assignment (noise, buckets, m_min
    suppression) directly degrade these features.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dtfl.attack.base import Adversary
from dtfl.attack.observation import DeviceObservation, ObservationFeaturizer

__all__ = ["L1FewShotAttacker"]


class L1FewShotAttacker(Adversary):
    """Random-forest classifier over observable tier-sequence features."""

    def __init__(
        self,
        featurizer: ObservationFeaturizer,
        min_observations: int = 5,
        n_estimators: int = 200,
        random_state: int = 0,
    ):
        """
        Parameters
        ----------
        featurizer:
            Turns observations into feature vectors (the only observable channel).
        min_observations:
            Devices seen fewer than this many rounds are predicted at the prior
            (too little signal); keeps the attack honest about low-information
            devices rather than guessing from one observation.
        """
        self._featurizer = featurizer
        self._min_obs = min_observations
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False
        self._majority_class: int | None = None

    def fit(
        self,
        seed_observations: list[DeviceObservation],
        seed_labels: np.ndarray,
    ) -> None:
        seed_labels = np.asarray(seed_labels)
        # Keep only seed devices with enough observations to carry signal.
        keep = [
            i
            for i, o in enumerate(seed_observations)
            if o.num_observations >= self._min_obs
        ]
        self._majority_class = int(np.bincount(seed_labels).argmax())
        if len(keep) < 2 or len(np.unique(seed_labels[keep])) < 2:
            # Not enough labeled signal to train; fall back to predicting prior.
            self._fitted = False
            return
        X = self._featurizer.featurize_many([seed_observations[i] for i in keep])
        y = seed_labels[keep]
        self._clf.fit(X, y)
        self._fitted = True

    def predict(self, query_observations: list[DeviceObservation]) -> np.ndarray:
        n = len(query_observations)
        if self._majority_class is None:
            raise RuntimeError("attacker must be fit before predict")
        preds = np.full(n, self._majority_class, dtype=np.int64)
        if not self._fitted:
            return preds  # untrained: predict the prior everywhere
        # Predict only for devices with enough observations; others stay at prior.
        rich_idx = [i for i, o in enumerate(query_observations) if o.num_observations >= self._min_obs]
        if rich_idx:
            X = self._featurizer.featurize_many([query_observations[i] for i in rich_idx])
            rich_preds = self._clf.predict(X)
            for j, i in enumerate(rich_idx):
                preds[i] = int(rich_preds[j])
        return preds
