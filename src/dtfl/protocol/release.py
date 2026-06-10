"""Release decision and transcript emission.

This is the boundary where latent/protocol state becomes the LEGAL TRANSCRIPT.
A tier releases its aggregate iff BOTH (preamble Section 5):

    1. n_{k,r} = |A_{k,r}| >= m_min                  (anonymity / privacy floor)
    2. unmask responders >= t_k                      (reconstruction feasible)

Otherwise the tier is SUPPRESSED (no aggregate, but a flag is still visible).

The emitted ``TierRecord`` carries ONLY whitelisted fields. Everything that went
into the decision -- device identities, true latencies, the roster, the active
indices -- stays on this side of the boundary. This module is where a careless
edit could leak ground truth into the transcript, so the record is constructed
explicitly field-by-field from non-identifying quantities.

The release-time bucket is computed here from the (latent) completion timeline,
coarsened to the configured granularity. The granularity is a DEFENSE KNOB; this
module just applies whatever bucketing function it is given (default: identity to
a single per-round bucket when no bucketer is supplied, i.e. maximally coarse).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dtfl.protocol.dropout import DropoutOutcome
from dtfl.types import RoundDeadlines, TierFlag, TierRecord

__all__ = ["ReleaseDecision", "decide_release", "emit_record"]


@dataclass(frozen=True)
class ReleaseDecision:
    """Outcome of the release gate for one tier session. Internal (not transcript).

    ``released`` is the final flag; ``reason`` explains a suppression for logging
    and for the predicted-vs-empirical release calibration plot (it is NOT part
    of the transcript -- the server only sees released/suppressed).
    """

    released: bool
    active_count: int
    unmask_count: int
    threshold: int
    m_min: int
    reason: str  # "ok" | "below_m_min" | "reconstruction_failed"


def decide_release(
    outcome: DropoutOutcome,
    m_min: int,
    threshold: int,
) -> ReleaseDecision:
    """Apply the two-condition release gate to a tier's dropout outcome."""
    n_k = outcome.active_count
    u_k = outcome.unmask_count

    if n_k < m_min:
        return ReleaseDecision(
            released=False,
            active_count=n_k,
            unmask_count=u_k,
            threshold=threshold,
            m_min=m_min,
            reason="below_m_min",
        )
    if u_k < threshold:
        return ReleaseDecision(
            released=False,
            active_count=n_k,
            unmask_count=u_k,
            threshold=threshold,
            m_min=m_min,
            reason="reconstruction_failed",
        )
    return ReleaseDecision(
        released=True,
        active_count=n_k,
        unmask_count=u_k,
        threshold=threshold,
        m_min=m_min,
        reason="ok",
    )


def emit_record(
    round_index: int,
    tier_index: int,
    decision: ReleaseDecision,
    deadlines: RoundDeadlines,
    secure_sum: np.ndarray | None,
    release_bucket: int | None,
    reveal_count: bool = True,
) -> TierRecord:
    """Construct the legal transcript record for one tier session.

    Every field is set explicitly from a non-identifying quantity. There is no
    code path here by which a device identity, true latency, or roster size can
    enter the record.

    Parameters
    ----------
    reveal_count:
        Whether the active count is revealed (preamble default: True). When a
        count-hiding defense is active this is False and ``count`` is None;
        note that hiding counts then biases the merge weight (handled in merge.py
        / measured in metrics/merge_error.py).
    secure_sum:
        The released aggregate, or None if suppressed. In Steps 0-2 this is a
        synthetic vector; the attacker treats it as opaque metadata regardless.
    release_bucket:
        Coarse release-time bucket, or None if suppressed before timing is set.
    """
    if decision.released:
        flag = TierFlag.RELEASED
        count = decision.active_count if reveal_count else None
        emitted_sum = secure_sum
        bucket = release_bucket
    else:
        flag = TierFlag.SUPPRESSED
        # A suppressed tier reveals only the flag (and optionally a coarse
        # suppression time). It does NOT reveal the count or the sum: those would
        # leak how-close-to-release a small/failed tier was.
        count = None
        emitted_sum = None
        bucket = release_bucket  # may be a coarse suppression-time bucket or None

    return TierRecord(
        round_index=round_index,
        tier_index=tier_index,
        flag=flag,
        count=count,
        release_bucket=bucket,
        secure_sum=emitted_sum,
        deadlines=deadlines,
    )


# A default maximally-coarse bucketer: everything in a round maps to bucket 0.
# Real bucketing (a defense knob) is supplied by dtfl.defense.timing_bucket later.
def coarsest_bucketer(_release_time: float, _deadlines: RoundDeadlines) -> int:
    """Map any release time to a single bucket (no timing signal). Safe default."""
    return 0


BucketerFn = Callable[[float, RoundDeadlines], int]
