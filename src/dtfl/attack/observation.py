"""What an attacker can observe about a device.

The threat model: pseudonyms are fresh per round, so the server cannot link
participations into per-device trajectories from the transcript alone. The L1
adversary's power comes from a different, realistic source -- it CONTROLS or
OBSERVES a set of devices (sybils / colluding clients / a surveilled target).
For a device it observes, the attacker can see WHICH TIER that device landed in
each round, because tier assignment is a function of the device's own behavior
that an observer co-located with it can read off (the device announces, or the
observer times, the tier it joined).

What the attacker is NOT given is the device's latent CAPABILITY CLASS. That is
the hidden attribute (Goal A). The attack tests whether the observable
tier-sequence leaks the hidden class.

This module is the SINGLE place that defines "observable about a device". It
deliberately produces features ONLY from:
  - the per-round tier a device occupied (observable to someone watching it), and
  - the transcript's per-round visible structure (buckets, counts, flags).
It never reads the device's capability class, mu, or true latency.

DESIGN BOUNDARY: ``DeviceObservation`` carries an observed tier sequence. It is
constructed by the experiment harness from the simulation's latent device->tier
record (which an attacker watching the device would equivalently obtain). The
class label is passed SEPARATELY and only for the labeled seed / for evaluation;
it is never inside the observation object the classifier reads at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dtfl.transcript.store import TranscriptView
from dtfl.types import TierFlag

__all__ = ["DeviceObservation", "build_observations", "ObservationFeaturizer"]


@dataclass(frozen=True)
class DeviceObservation:
    """An attacker's observation of one device over the horizon.

    ``tier_sequence``: list of (round_index, tier_index) the device was seen in.
    A device that missed a round or was suppressed-out simply has no entry for
    that round (the observer sees it not participate / not release).

    This object contains NO latent attribute. The capability label lives outside,
    keyed by device id, available only for the seed and for evaluation.
    """

    device_id: int
    tier_sequence: list[tuple[int, int]]  # (round, tier)

    @property
    def num_observations(self) -> int:
        return len(self.tier_sequence)


class ObservationFeaturizer:
    """Turns a DeviceObservation (+ the public transcript) into a feature vector.

    Features are derived ONLY from observable quantities:
      - tier-occupancy histogram (fraction of participations in each tier),
      - mean and std of occupied tier index,
      - transition frequencies between consecutive tiers,
      - participation rate (observations / rounds seen),
      - bucket statistics for the tiers the device occupied (from the transcript).

    The bucket statistics tie the per-device signal to the transcript's visible
    timing channel, so coarsening buckets (a defense) directly weakens features.
    """

    def __init__(self, num_tiers: int, num_rounds: int, view: TranscriptView | None = None):
        self.num_tiers = num_tiers
        self.num_rounds = num_rounds
        self._view = view
        # Cache per-(round,tier) bucket from the transcript, if provided.
        self._bucket_lookup: dict[tuple[int, int], int | None] = {}
        if view is not None:
            for rec in view.all():
                self._bucket_lookup[(rec.round_index, rec.tier_index)] = rec.release_bucket

    @property
    def feature_dim(self) -> int:
        # occupancy(K) + [mean_tier, std_tier, participation_rate] + transitions(K*K)
        # + [mean_bucket, std_bucket]
        return self.num_tiers + 3 + self.num_tiers * self.num_tiers + 2

    def featurize(self, obs: DeviceObservation) -> np.ndarray:
        K = self.num_tiers
        feat = np.zeros(self.feature_dim, dtype=np.float64)
        seq = [t for (_, t) in obs.tier_sequence]

        if not seq:
            return feat  # all-zero for a never-observed device

        tiers = np.asarray(seq, dtype=np.int64)

        # --- occupancy histogram ---
        occ = np.bincount(tiers, minlength=K).astype(np.float64)
        occ_sum = occ.sum()
        if occ_sum > 0:
            occ /= occ_sum
        feat[:K] = occ

        # --- summary stats of occupied tier index ---
        feat[K] = float(tiers.mean())
        feat[K + 1] = float(tiers.std())
        feat[K + 2] = obs.num_observations / max(1, self.num_rounds)

        # --- transition frequencies between consecutive observations ---
        base = K + 3
        if tiers.size >= 2:
            trans = np.zeros((K, K), dtype=np.float64)
            for a, b in zip(tiers[:-1], tiers[1:], strict=True):
                trans[a, b] += 1.0
            tsum = trans.sum()
            if tsum > 0:
                trans /= tsum
            feat[base : base + K * K] = trans.ravel()

        # --- bucket statistics from the transcript for occupied tier-rounds ---
        bidx = base + K * K
        if self._view is not None:
            buckets = []
            for r, t in obs.tier_sequence:
                b = self._bucket_lookup.get((r, t))
                if b is not None:
                    buckets.append(b)
            if buckets:
                feat[bidx] = float(np.mean(buckets))
                feat[bidx + 1] = float(np.std(buckets))

        return feat

    def featurize_many(self, observations: list[DeviceObservation]) -> np.ndarray:
        if not observations:
            return np.zeros((0, self.feature_dim), dtype=np.float64)
        return np.vstack([self.featurize(o) for o in observations])


def build_observations(
    device_tier_records: dict[int, list[tuple[int, int]]],
) -> list[DeviceObservation]:
    """Build observations from a {device_id: [(round, tier), ...]} mapping.

    The harness constructs this mapping from the simulation's latent device->tier
    membership -- which is exactly what an attacker observing those devices would
    obtain. No capability labels enter here.
    """
    return [
        DeviceObservation(device_id=did, tier_sequence=sorted(seq))
        for did, seq in device_tier_records.items()
    ]
