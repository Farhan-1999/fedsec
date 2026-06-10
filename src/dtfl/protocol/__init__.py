"""dtfl.protocol: consumes latent draws, produces the legal transcript."""
from dtfl.protocol.dropout import DropoutOutcome, DropoutRates, apply_dropout
from dtfl.protocol.merge import (
    TierContribution,
    apply_server_update,
    size_weighted_merge,
    weighted_merge,
)
from dtfl.protocol.phases import PhaseCutoffs, PhaseDeltas
from dtfl.protocol.release import ReleaseDecision, decide_release, emit_record
from dtfl.protocol.threshold import reconstruction_threshold, safe_coupling_holds
from dtfl.protocol.tiering import MISSED, assign_tiers, tier_rosters

__all__ = [
    "assign_tiers", "tier_rosters", "MISSED",
    "PhaseCutoffs", "PhaseDeltas",
    "DropoutRates", "DropoutOutcome", "apply_dropout",
    "reconstruction_threshold", "safe_coupling_holds",
    "ReleaseDecision", "decide_release", "emit_record",
    "TierContribution", "size_weighted_merge", "weighted_merge", "apply_server_update",
]
