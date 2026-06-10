"""dtfl.metrics: what the paper reports."""
from dtfl.metrics.advantage import AdvantageResult, capability_advantage
from dtfl.metrics.anonymity import (
    AnonymityStats,
    anonymity_stats,
    equivalence_class_sizes,
    signature_up_to,
)
from dtfl.metrics.linkability import (
    LinkabilityCurve,
    linkability_curve,
    mean_linkability_at,
)

__all__ = [
    "AdvantageResult", "capability_advantage",
    "AnonymityStats", "anonymity_stats", "equivalence_class_sizes", "signature_up_to",
    "LinkabilityCurve", "linkability_curve", "mean_linkability_at",
]
