"""The transcript store: append-only log of TierRecords.

This is the attacker's SOLE input. The protocol layer appends records as rounds
complete; the attack layer is handed a (read-only) view and may query only
through this API. Every method returns TierRecords or quantities derived purely
from their whitelisted fields -- there is no accessor that reaches latent state,
by construction, because the store only ever holds TierRecords.

Two access roles, separated to make misuse obvious:
- the SIMULATION side calls ``append`` (write),
- the ATTACK side receives a ``TranscriptView`` (read-only) and cannot append.

The view is what the separation guarantee rests on at runtime: the attacker is
constructed with a TranscriptView, never with the latent store or the writable
TranscriptStore.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from dtfl.types import TierFlag, TierRecord

__all__ = ["TranscriptStore", "TranscriptView"]


class TranscriptView:
    """Read-only window onto a transcript. The attacker's only handle.

    Wraps an underlying list of records by reference (no copy) but exposes no
    mutation. All queries are over whitelisted TierRecord fields.
    """

    def __init__(self, records: Sequence[TierRecord]):
        self._records = records

    # --- basic access ---

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[TierRecord]:
        return iter(self._records)

    def all(self) -> list[TierRecord]:
        """All records, in append (round) order."""
        return list(self._records)

    def released(self) -> list[TierRecord]:
        """Only records for tiers that released an aggregate."""
        return [r for r in self._records if r.flag is TierFlag.RELEASED]

    # --- per-round / per-tier slices ---

    def for_round(self, round_index: int) -> list[TierRecord]:
        return [r for r in self._records if r.round_index == round_index]

    def for_tier(self, tier_index: int) -> list[TierRecord]:
        return [r for r in self._records if r.tier_index == tier_index]

    def rounds(self) -> list[int]:
        """Sorted unique round indices present in the transcript."""
        return sorted({r.round_index for r in self._records})

    # --- derived aggregate signals (still only whitelisted fields) ---

    def tier_counts(self, round_index: int) -> dict[int, int | None]:
        """Map tier_index -> revealed active count for a round.

        Count is None for suppressed tiers (and for released tiers under a
        count-hiding defense). This is exactly the count vector the controller
        and the attacker's count-flow linker operate on.
        """
        return {r.tier_index: r.count for r in self.for_round(round_index)}

    def release_buckets(self, round_index: int) -> dict[int, int | None]:
        """Map tier_index -> coarse release bucket for a round."""
        return {r.tier_index: r.release_bucket for r in self.for_round(round_index)}

    def release_flags(self, round_index: int) -> dict[int, TierFlag]:
        """Map tier_index -> released/suppressed flag for a round."""
        return {r.tier_index: r.flag for r in self.for_round(round_index)}

    def signature(self, round_index: int, tier_index: int) -> tuple:
        """The visible (tier, release-bucket) signature for one tier-round.

        This is the atom the linkability attack builds equivalence classes from
        (adversary spec: sigma_i = ((k, b))_r). Returns a hashable tuple of only
        visible fields.
        """
        recs = [
            r
            for r in self._records
            if r.round_index == round_index and r.tier_index == tier_index
        ]
        if not recs:
            return ()
        r = recs[0]
        return (r.tier_index, r.release_bucket, r.flag.value)


class TranscriptStore:
    """Writable, append-only transcript. Held by the SIMULATION side only.

    Hand ``view()`` to the attacker; never hand the store itself.
    """

    def __init__(self) -> None:
        self._records: list[TierRecord] = []

    def append(self, record: TierRecord) -> None:
        self._records.append(record)

    def extend(self, records: Sequence[TierRecord]) -> None:
        self._records.extend(records)

    def view(self) -> TranscriptView:
        """Return a read-only view for the attacker. Shares storage by reference.

        Because the view holds the same underlying list, records appended later
        become visible to an already-issued view -- matching an online adversary
        that observes the transcript as it grows. Snapshot with ``list(view.all())``
        if a frozen copy is needed.
        """
        return TranscriptView(self._records)

    def __len__(self) -> int:
        return len(self._records)
