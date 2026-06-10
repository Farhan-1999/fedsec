"""Internal secure-aggregation phase cutoffs for a tier session.

Secure aggregation is interactive: join -> share-distribution -> masked-upload
-> unmasking. Each has a sub-deadline inside the tier's wall-clock window. The
dropout-safety rule (dropout doc): the server must NOT begin unmasking until
after D_mask + grace, or it risks misclassifying slow-but-arriving clients as
dropouts.

    D_join  <  D_share  <  D_mask (= the broadcast tier deadline)  <  D_unmask
                                      |
                                      +-- masked uploads accepted until D_mask + grace

In Steps 0-2 these cutoffs are bookkeeping that the dropout model and the timing
controller consume; they do not yet drive a real interactive protocol (that is
the Step-6 calibration). They are defined now so the dropout phases and the
release-time bucket are anchored to a concrete timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PhaseCutoffs", "PhaseDeltas"]


@dataclass(frozen=True)
class PhaseDeltas:
    """Wall-clock gaps allocated to each interactive phase (seconds).

    The controller is responsible for ensuring the tier window is wide enough to
    fit these. ``grace`` is the late-arrival tolerance after the masked-upload
    cutoff before unmasking may begin.
    """

    setup: float = 1.0  # join + share distribution span (D_mask - D_share - D_join slack)
    upload: float = 0.5  # share-cutoff to mask-cutoff span
    unmask: float = 1.0  # mask-cutoff to unmask-completion span
    grace: float = 0.25  # late-arrival window after D_mask

    def __post_init__(self) -> None:
        for name, v in vars(self).items():
            if v < 0:
                raise ValueError(f"phase delta {name} must be >= 0, got {v}")


@dataclass(frozen=True)
class PhaseCutoffs:
    """Concrete per-tier-session cutoffs derived from the tier deadline D_mask."""

    d_join: float
    d_share: float
    d_mask: float
    d_unmask: float
    grace: float

    @property
    def mask_accept_until(self) -> float:
        """Latest a masked upload is accepted as active."""
        return self.d_mask + self.grace

    @classmethod
    def from_deadline(cls, d_mask: float, deltas: PhaseDeltas) -> PhaseCutoffs:
        """Build the cutoff schedule by working backward/forward from D_mask."""
        d_share = d_mask - deltas.upload
        d_join = d_share - deltas.setup
        d_unmask = d_mask + deltas.grace + deltas.unmask
        if d_join < 0:
            # The tier deadline is too tight to fit the interactive phases; the
            # controller should widen it. We clamp join to 0 and let the dropout
            # model reflect the squeeze rather than crash.
            d_join = 0.0
        return cls(
            d_join=d_join,
            d_share=d_share,
            d_mask=d_mask,
            d_unmask=d_unmask,
            grace=deltas.grace,
        )
