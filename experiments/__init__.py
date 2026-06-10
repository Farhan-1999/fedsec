"""dtfl.attack: adversaries. Imports transcript/, metrics/, types, rng ONLY."""
from dtfl.attack.base import Adversary
from dtfl.attack.l0_unsupervised import L0UnsupervisedAttacker
from dtfl.attack.l1_fewshot import L1FewShotAttacker
from dtfl.attack.l2_prior import L2PriorAttacker, ModelKnowledge
from dtfl.attack.l3_omniscient import L3OmniscientAttacker
from dtfl.attack.observation import (
    DeviceObservation,
    ObservationFeaturizer,
    build_observations,
)

__all__ = [
    "Adversary",
    "L0UnsupervisedAttacker", "L1FewShotAttacker",
    "L2PriorAttacker", "ModelKnowledge", "L3OmniscientAttacker",
    "DeviceObservation", "ObservationFeaturizer", "build_observations",
]
