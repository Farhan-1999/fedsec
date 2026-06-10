"""TierRecord construction helpers and serialization.

The TierRecord dataclass itself lives in dtfl.types (so the separation tests can
inspect it without importing the protocol layer). This module provides:
- a thin bridge that turns a release decision + a bucketer into a record, and
- JSON serialization so seeded transcripts can be saved and reloaded (an
  artifact-reproducibility requirement: same seed -> same transcript on disk).

Serialization deliberately stores ONLY whitelisted fields. A reloaded transcript
is therefore provably free of latent state -- you could publish the transcript
files and they would leak nothing the running server didn't already see.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dtfl.types import RoundDeadlines, TierFlag, TierRecord

__all__ = ["save_transcript", "load_transcript", "record_to_dict", "record_from_dict"]


def record_to_dict(record: TierRecord) -> dict:
    """Serialize a TierRecord to a JSON-safe dict (whitelisted fields only)."""
    return {
        "round_index": record.round_index,
        "tier_index": record.tier_index,
        "flag": record.flag.value,
        "count": record.count,
        "release_bucket": record.release_bucket,
        # Store the secure sum as a list (or null). In Steps 0-2 it is synthetic;
        # the attacker treats it as opaque, so we keep it for fidelity but it is
        # never required to reconstruct the attack's inputs.
        "secure_sum": (
            np.asarray(record.secure_sum).tolist() if record.secure_sum is not None else None
        ),
        "deadlines": {
            "round_index": record.deadlines.round_index,
            "cutoffs": list(record.deadlines.cutoffs),
        },
    }


def record_from_dict(d: dict) -> TierRecord:
    """Inverse of ``record_to_dict``."""
    return TierRecord(
        round_index=d["round_index"],
        tier_index=d["tier_index"],
        flag=TierFlag(d["flag"]),
        count=d["count"],
        release_bucket=d["release_bucket"],
        secure_sum=(
            np.asarray(d["secure_sum"], dtype=np.float64) if d["secure_sum"] is not None else None
        ),
        deadlines=RoundDeadlines(
            round_index=d["deadlines"]["round_index"],
            cutoffs=tuple(d["deadlines"]["cutoffs"]),
        ),
    )


def save_transcript(records: list[TierRecord], path: str | Path) -> None:
    """Write a transcript to JSON. Only whitelisted fields are persisted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record_to_dict(r) for r in records]
    path.write_text(json.dumps(payload, separators=(",", ":")))


def load_transcript(path: str | Path) -> list[TierRecord]:
    """Read a transcript previously written by ``save_transcript``."""
    path = Path(path)
    payload = json.loads(path.read_text())
    return [record_from_dict(d) for d in payload]
