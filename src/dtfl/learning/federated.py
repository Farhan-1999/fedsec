"""Federated training loop with real model updates.

Bridges the privacy/timing simulator and a real model. The engine still decides
WHO participates in WHICH tier each round (the timing + suppression + dropout
machinery, unchanged), but the per-client update is now a real local-SGD delta,
tier sums are real sums of deltas, the merge is the FedAvg-equivalent size-
weighted rule, and the global model is updated and evaluated each round. The
output adds the missing UTILITY axis: validation accuracy vs round and vs virtual
wall-clock time (time-to-accuracy / round-to-accuracy).

Crucially this reuses the SAME tiering/release logic, so the privacy transcript
produced here is identical in structure to the metadata-only runs -- the attack
and linkability metrics apply unchanged. The only thing that becomes "real" is
the content of the tier sums and therefore the learning curve. The privacy
results do not depend on update contents (the attack reads metadata), so this
layer adds utility measurement without disturbing the privacy story.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import time

import numpy as np

from dtfl.learning.datasets import iid_shard
from dtfl.learning.model import Dataset, FLModel
from dtfl.protocol.dropout import apply_dropout
from dtfl.protocol.merge import TierContribution, apply_server_update, size_weighted_merge
from dtfl.protocol.release import decide_release
from dtfl.protocol.threshold import reconstruction_threshold
from dtfl.protocol.tiering import assign_tiers, tier_rosters
from dtfl.latent.latency import draw_completion_times
from dtfl.rng import RngHub
from dtfl.sim.engine import Engine, EngineConfig
from dtfl.sim.round import RoundConfig

__all__ = ["FedTrainConfig", "FedTrainResult", "federated_train"]


@dataclass
class FedTrainConfig:
    local_epochs: int = 1
    local_lr: float = 0.5
    local_batch: int = 32
    server_lr: float = 1.0
    momentum: float = 0.0


@dataclass
class FedTrainResult:
    val_accuracy: list[float]  # per round
    val_loss: list[float]
    virtual_time: list[float]  # cumulative SIMULATED round budget (deadlines)
    participation: list[int]
    wall_time: list[float] = field(default_factory=list)  # cumulative MEASURED seconds
    round_time: list[float] = field(default_factory=list)  # cumulative straggler-aware time

    def round_to_accuracy(self, target: float) -> int | None:
        for r, a in enumerate(self.val_accuracy):
            if a >= target:
                return r
        return None

    def straggler_time_to_accuracy(self, target: float) -> float | None:
        for t, a in zip(self.round_time, self.val_accuracy, strict=True):
            if a >= target:
                return t
        return None

    def time_to_accuracy(self, target: float) -> float | None:
        for t, a in zip(self.virtual_time, self.val_accuracy, strict=True):
            if a >= target:
                return t
        return None


def federated_train(
    model: FLModel,
    train: Dataset,
    val: Dataset,
    engine: Engine,
    deadlines_cutoffs: tuple[float, ...],
    train_cfg: FedTrainConfig,
    round_config: RoundConfig,
    *,
    base_seed: int = 7,
    tier_selector=None,
) -> FedTrainResult:
    """Run real federated training under the engine's timing/tier dynamics.

    The engine instance supplies the population (its latent mu/classes/availability
    and drift). We re-derive per-round participation here using the same tiering +
    dropout logic so we can attach a real client dataset shard to each device and
    compute real deltas. Client shards are IID (preamble Section 2).

    tier_selector: optional callable (round_index, num_tiers, rng) -> iterable
    of tier indices to TRAIN this round. Default None trains all tiers (our
    framework / FedAvg). A TiFL-style baseline passes a selector that returns a
    single tier per round.
    """
    hub = RngHub(seed=base_seed)
    # Pull population state off the engine (latent side; this is experimenter code).
    mu = engine._mu
    classes = engine._classes
    avail = engine._availability
    lcfg = engine._lcfg
    N = mu.shape[0]
    num_tiers = len(deadlines_cutoffs)

    # IID shards: one per device.
    shards = iid_shard(train, N, hub.stream("shard"))

    from dtfl.types import RoundDeadlines
    deadlines = RoundDeadlines(round_index=0, cutoffs=tuple(deadlines_cutoffs))

    val_acc, val_loss, vtime, partic = [], [], [], []
    wtime = []
    rtime = []          # cumulative straggler-aware round time
    round_clock = 0.0
    wall_clock = 0.0
    clock = 0.0
    avail_rng = hub.stream("avail")

    num_rounds = engine._cfg.num_rounds
    for r in range(num_rounds):
        _t0 = time.perf_counter()
        child = hub.child(f"round.{r}")
        available_mask = avail_rng.random(N) < avail
        avail_ids = np.flatnonzero(available_mask)

        # completion times -> tier assignment (same logic as the metadata sim)
        tau = draw_completion_times(mu[avail_ids], lcfg, child.stream("latency"))
        tiers_local = assign_tiers(tau, deadlines)
        rosters_local = tier_rosters(tiers_local, num_tiers)
        # completion time of the slowest device in each tier (for round-time cost)
        tier_max_tau = []
        for _k in range(num_tiers):
            _rl = rosters_local[_k]
            tier_max_tau.append(float(tau[_rl].max()) if _rl.size > 0 else 0.0)

        contributions = []
        trained_tiers = []
        w_global = model.get_params()
        if tier_selector is None:
            selected = set(range(num_tiers))
        else:
            selected = set(tier_selector(r, num_tiers, child.stream("tiersel")))
        for k in range(num_tiers):
            if k not in selected:
                continue
            roster_local = rosters_local[k]
            if roster_local.size == 0:
                continue
            outcome = apply_dropout(roster_local.size, round_config.dropout_rates, child.stream(f"drop.{k}"))
            active_local = roster_local[outcome.active]
            active_ids = avail_ids[active_local]
            n_k = outcome.active_count

            t_k = reconstruction_threshold(n_k, round_config.m_min, round_config.dropout_rates)
            decision = decide_release(outcome, round_config.m_min, t_k)
            if not decision.released:
                continue

            # Real tier sum = sum of active clients' local deltas.
            tier_sum = np.zeros(model.dim, dtype=np.float64)
            for did in active_ids:
                model.set_params(w_global)  # each client starts from the global model
                delta = model.local_update(
                    shards[int(did)],
                    train_cfg.local_epochs, train_cfg.local_lr,
                    train_cfg.local_batch, child.stream(f"client.{did}"),
                )
                tier_sum += delta
            contributions.append(TierContribution(secure_sum=tier_sum, count=n_k))
            trained_tiers.append(k)

        model.set_params(w_global)  # restore before applying merged update
        merged = size_weighted_merge(contributions)
        w_next, _ = apply_server_update(
            w_global, merged, lr=train_cfg.server_lr, momentum=train_cfg.momentum
        )
        model.set_params(w_next)

        loss, acc = model.evaluate(val)
        val_acc.append(acc); val_loss.append(loss)
        if tier_selector is not None and hasattr(tier_selector, "update"):
            tier_selector.update(selected, acc, r)
        total_part = sum(c.count for c in contributions)
        partic.append(total_part)
        clock += float(deadlines_cutoffs[-1])
        vtime.append(clock)
        # straggler-aware round time: the round ends when the slowest device
        # whose tier actually trained has finished. Frameworks that skip slow
        # tiers pay less per round. If nothing trained, charge the fastest tier.
        if trained_tiers:
            round_cost = max(tier_max_tau[k] for k in trained_tiers)
        else:
            round_cost = min((t for t in tier_max_tau if t > 0), default=0.0)
        round_clock += round_cost
        rtime.append(round_clock)
        wall_clock += time.perf_counter() - _t0
        wtime.append(wall_clock)

    return FedTrainResult(
        val_accuracy=val_acc, val_loss=val_loss, virtual_time=vtime,
        participation=partic, wall_time=wtime, round_time=rtime
    )
