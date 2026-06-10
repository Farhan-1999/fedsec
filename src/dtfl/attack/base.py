"""Adversary interface.

Every attacker implements ``fit`` (using a labeled seed of observations) and
``predict`` (capability-class predictions for query observations). Attackers are
constructed and called with ONLY:
  - DeviceObservation objects (observable tier sequences),
  - a TranscriptView (the legal transcript),
  - capability labels for the SEED devices (the L1 auxiliary knowledge).

They never receive latent state for query devices. The labels supplied to
``fit`` are the few-shot seed the L1 adversary is assumed to possess; labels are
NEVER supplied to ``predict``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dtfl.attack.observation import DeviceObservation

__all__ = ["Adversary"]


class Adversary(ABC):
    """Base class for capability-inference adversaries."""

    @abstractmethod
    def fit(
        self,
        seed_observations: list[DeviceObservation],
        seed_labels: np.ndarray,
    ) -> None:
        """Learn from a labeled seed of (observation, true class) pairs."""

    @abstractmethod
    def predict(self, query_observations: list[DeviceObservation]) -> np.ndarray:
        """Predict capability classes for query observations (no labels given)."""
