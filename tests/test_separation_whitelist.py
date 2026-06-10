"""Separation test 2: the transcript record exposes only legal fields.

A positive whitelist, not a blacklist. We assert that the data structure the
attacker is handed (``TierRecord``) carries exactly the field set permitted by
the adversary spec -- no more. If someone later adds a latent field to the
record (e.g. ``true_latency`` "just for debugging"), this test fails loudly.

Pairing this with the import-graph test gives two independent guarantees:
- the attacker cannot *import* ground-truth code (test 1), and
- the object it *does* receive carries no ground-truth fields (test 2).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dtfl.types import RoundDeadlines, TierFlag, TierRecord

pytestmark = pytest.mark.separation

# The exact whitelist from the adversary spec, Section 2.
LEGAL_FIELDS = frozenset(
    {
        "round_index",
        "tier_index",
        "flag",
        "count",
        "release_bucket",
        "secure_sum",
        "deadlines",
    }
)

# Field names that would constitute a leak if they ever appeared on the record.
FORBIDDEN_SUBSTRINGS = (
    "capability",
    "class",
    "theta",
    "mu",
    "true_",
    "latency",
    "device_id",
    "roster",  # only the active count is legal, never the roster size
    "availability",
)


def _make_record() -> TierRecord:
    return TierRecord(
        round_index=0,
        tier_index=1,
        flag=TierFlag.RELEASED,
        count=42,
        release_bucket=3,
        secure_sum=np.zeros(4, dtype=np.float64),
        deadlines=RoundDeadlines(round_index=0, cutoffs=(1.0, 2.0, 3.0)),
    )


def test_tier_record_fields_match_whitelist_exactly():
    """The dataclass fields equal the legal whitelist -- exactly."""
    actual = {f.name for f in dataclasses.fields(TierRecord)}
    assert actual == LEGAL_FIELDS, (
        "TierRecord fields drifted from the legal observation set.\n"
        f"  unexpected (possible leak): {sorted(actual - LEGAL_FIELDS)}\n"
        f"  missing: {sorted(LEGAL_FIELDS - actual)}\n"
        "Changing the visible field set is a threat-model change: update the "
        "adversary spec and this whitelist together, or remove the field."
    )


def test_record_self_reported_whitelist_matches():
    """The record's own ``visible_field_names`` agrees with the test whitelist."""
    rec = _make_record()
    assert rec.visible_field_names() == LEGAL_FIELDS


def test_no_forbidden_field_names_present():
    """No field name hints at a latent/ground-truth quantity."""
    actual = {f.name for f in dataclasses.fields(TierRecord)}
    offenders = sorted(
        name
        for name in actual
        for bad in FORBIDDEN_SUBSTRINGS
        if bad in name.lower()
    )
    assert not offenders, (
        f"TierRecord has field name(s) suggesting a ground-truth leak: {offenders}."
    )
