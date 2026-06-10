"""One simulation round, end to end.

Wires the three data layers for a single round:

  broadcast deadlines
    -> draw completion times (latent)
    -> self-assign to tiers (protocol)
    -> per tier: roster -> dropout -> active set
    -> threshold + release gate
    -> compute release time, coarsen to bucket
    -> emit TierRecord (transcript)
    -> collect contributions, merge into one global update

The function returns both the emitted records (the legal transcript) and a
``RoundLatentLog`` of ground-truth quantities (true classes per tier, true
completion times, release decisions with reasons). The latent log is for the
EXPERIMENTER -- it drives the SNR diagnostic, the predicted-vs-empirical release
calibration, and ground-truth attack evaluation. It is NEVER handed to the
attacker; the engine keeps it on the latent side of the boundary.

In Steps 0-2 the secure sum is synthetic (drawn here), because the attack reads
metadata, not sum contents. Step 4 replaces the synthetic draw with real
aggregated model updates without changing this control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dtfl.latent.config import LatentConfig
from dtfl.latent.latency import draw_completion_times
from dtfl.protocol.dropout import DropoutRates, apply_dropout
from dtfl.protocol.merge import TierContribution, size_weighted_merge
from dtfl.protocol.release import decide_release, emit_record
from dtfl.protocol.threshold import reconstruction_threshold
from dtfl.protocol.tiering import assign_tiers, tier_rosters
from dtfl.transcript.bucket import Bucketer, single_bucket
from dtfl.types import RoundDeadlines, TierRecord

__all__ = ["RoundConfig", "RoundLatentLog", "RoundResult", "run_round"]


@dataclass(frozen=True)
class RoundConfig:
    """Per-round protocol parameters (the parts the engine holds fixed or varies)."""

    m_min: int = 16
    dropout_rates: DropoutRates = field(default_factory=DropoutRates)
    delta: float = 1e-3  # Hoeffding failure target for threshold
    update_dim: int = 16  # dimension of the (synthetic) update vectors in Steps 0-2
    reveal_count: bool = True  # preamble default; False under count-hiding defense


@dataclass
class RoundLatentLog:
    """Ground-truth, per-round bookkeeping. EXPERIMENTER-ONLY, never to attacker.

    Index-aligned per tier (length = num_tiers). Entries for tiers that had no
    roster are still present (empty arrays / None) so downstream code can index
    by tier uniformly.
    """

    round_index: int
    # true capability classes of the ACTIVE devices in each tier
    active_classes: list[np.ndarray]
    # true completion times of the active devices in each tier
    active_taus: list[np.ndarray]
    # device ids active in each tier (latent identity)
    active_device_ids: list[np.ndarray]
    # release decision reason per tier ("ok"/"below_m_min"/"reconstruction_failed"/"empty")
    release_reasons: list[str]
    # roster size per tier (latent: never revealed)
    roster_sizes: list[int]
    # missed-deadline dropout count this round
    missed_count: int


@dataclass
class RoundResult:
    """Everything one round produces."""

    records: list[TierRecord]  # the legal transcript for this round
    latent_log: RoundLatentLog  # experimenter-only ground truth
    merged_update: np.ndarray | None  # size-weighted merge over successful tiers
    total_participation: int  # n_r = sum of released tier counts


def run_round(
    round_index: int,
    deadlines: RoundDeadlines,
    mu: np.ndarray,
    classes: np.ndarray,
    available_mask: np.ndarray,
    config: LatentConfig,
    round_config: RoundConfig,
    latency_rng: np.random.Generator,
    protocol_rng: np.random.Generator,
    bucketer: Bucketer = single_bucket,
    shared_log_offset: float = 0.0,
) -> RoundResult:
    """Execute one round and return records, latent log, and the merged update.

    Parameters
    ----------
    mu, classes:
        Full-population latent base log-latencies and capability classes (length N).
    available_mask:
        Boolean array (length N); only available devices participate this round.
    latency_rng, protocol_rng:
        Independent streams for the latency draw and the dropout draws.
    bucketer:
        Maps a release time + deadlines to a coarse bucket (a defense knob).
    shared_log_offset:
        Drift offset applied to all completion times this round.
    """
    num_tiers = deadlines.num_tiers
    avail_ids = np.flatnonzero(available_mask)

    # --- latent: completion times for available devices only ---
    tau_avail = draw_completion_times(
        mu[avail_ids], config, latency_rng, shared_log_offset=shared_log_offset
    )

    # --- protocol: tier self-assignment (indices are into avail_ids) ---
    tiers_local = assign_tiers(tau_avail, deadlines)
    rosters_local = tier_rosters(tiers_local, num_tiers)
    missed_count = int((tiers_local == -1).sum())

    records: list[TierRecord] = []
    contributions: list[TierContribution] = []
    active_classes: list[np.ndarray] = []
    active_taus: list[np.ndarray] = []
    active_device_ids: list[np.ndarray] = []
    release_reasons: list[str] = []
    roster_sizes: list[int] = []
    total_participation = 0

    for k in range(num_tiers):
        roster_local = rosters_local[k]  # indices into avail_ids
        roster_sizes.append(int(roster_local.size))

        if roster_local.size == 0:
            # No roster: emit nothing visible (no tier session ran).
            active_classes.append(np.empty(0, dtype=np.int64))
            active_taus.append(np.empty(0, dtype=np.float64))
            active_device_ids.append(np.empty(0, dtype=np.int64))
            release_reasons.append("empty")
            continue

        # --- dropout: winnow roster to active + unmask responders ---
        outcome = apply_dropout(roster_local.size, round_config.dropout_rates, protocol_rng)
        # Map roster-local active indices back to device ids and taus.
        active_local = roster_local[outcome.active]  # indices into avail_ids
        active_ids = avail_ids[active_local]  # global device ids
        active_tau = tau_avail[active_local]
        active_cls = classes[active_ids]

        active_classes.append(active_cls)
        active_taus.append(active_tau)
        active_device_ids.append(active_ids)

        n_k = outcome.active_count

        # --- threshold + release gate ---
        t_k = reconstruction_threshold(
            n_k, round_config.m_min, round_config.dropout_rates, delta=round_config.delta
        )
        decision = decide_release(outcome, round_config.m_min, t_k)
        release_reasons.append(decision.reason)

        # --- release time -> coarse bucket (only the bucket is visible) ---
        # Release time = when the last active device's masked upload completes;
        # use the max active tau as a simple, monotone proxy.
        release_time = float(active_tau.max()) if active_tau.size else float(deadlines.cutoffs[k])
        bucket = bucketer(release_time, deadlines)

        # --- synthetic secure sum (Steps 0-2): sum of per-device synthetic updates ---
        # Deterministic given (round, tier, device ids) so it is reproducible and
        # so FedAvg-equivalence checks are exact; contents are not used by the attack.
        if decision.released:
            # Build per-device synthetic updates seeded by device id (stable).
            updates = _synthetic_updates(active_ids, round_index, round_config.update_dim)
            secure_sum = updates.sum(axis=0)
            contributions.append(TierContribution(secure_sum=secure_sum, count=n_k))
            total_participation += n_k
        else:
            secure_sum = None

        record = emit_record(
            round_index=round_index,
            tier_index=k,
            decision=decision,
            deadlines=deadlines,
            secure_sum=secure_sum,
            release_bucket=bucket,
            reveal_count=round_config.reveal_count,
        )
        records.append(record)

    merged = size_weighted_merge(contributions)

    latent_log = RoundLatentLog(
        round_index=round_index,
        active_classes=active_classes,
        active_taus=active_taus,
        active_device_ids=active_device_ids,
        release_reasons=release_reasons,
        roster_sizes=roster_sizes,
        missed_count=missed_count,
    )

    return RoundResult(
        records=records,
        latent_log=latent_log,
        merged_update=merged,
        total_participation=total_participation,
    )


def _synthetic_updates(device_ids: np.ndarray, round_index: int, dim: int) -> np.ndarray:
    """Deterministic synthetic per-device update vectors (Steps 0-2 only).

    Seeded by (device_id, round) so the same device contributes reproducibly and
    FedAvg-equivalence is exact. Contents are irrelevant to the metadata attack;
    Step 4 replaces this with real local-training deltas.
    """
    out = np.empty((device_ids.size, dim), dtype=np.float64)
    for idx, did in enumerate(device_ids):
        gen = np.random.default_rng((int(did) << 20) ^ round_index)
        out[idx] = gen.normal(size=dim)
    return out
