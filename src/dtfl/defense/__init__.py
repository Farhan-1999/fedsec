"""dtfl.defense: privacy knobs that perturb the transcript."""
from dtfl.defense.config import BucketMode, CountMode, DefenseConfig
from dtfl.defense.count_noise import apply_count_defense, merge_weight_for
from dtfl.defense.min_size import deadline_quantiles_for, m_min_of
from dtfl.defense.padding import PaddingResult, apply_padding
from dtfl.defense.send_delay import apply_send_delay
from dtfl.defense.timing_bucket import bucketer_for

__all__ = [
    "DefenseConfig", "BucketMode", "CountMode",
    "apply_count_defense", "merge_weight_for",
    "apply_send_delay",
    "PaddingResult", "apply_padding",
    "deadline_quantiles_for", "m_min_of",
    "bucketer_for",
]
