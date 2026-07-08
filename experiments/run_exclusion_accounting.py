"""Client-exclusion accounting: how many clients are dropped, and why.

Compares, per round and in total, how many rostered-and-available clients end up
NOT contributing to the global update under two schemes at matched N, K, seed:

  ours  (tiered + m_min suppression): a client is excluded if its tier is
         SUPPRESSED (n_k < m_min) or lost to secure-aggregation DROPOUT.
  tifl  (speed-biased single-tier/round): a client is excluded if its tier is
         NOT SELECTED this round, or lost to DROPOUT.

We separate the causes so the comparison is honest:
  - selected_out : excluded because the scheme did not train that client's tier
                   (TiFL trains one tier/round; ours trains all released tiers)
  - suppressed    : excluded because the tier failed the m_min anonymity floor
  - dropout       : excluded by masking/unmask attrition (same mechanism for both)

The point: the two schemes drop clients for DIFFERENT reasons. TiFL discards
whole tiers every round by design (its speed bias); ours discards only the small
tiers the anonymity floor suppresses. This quantifies the participation each pays.

Runs through the identical engine / seed / data / model init as the utility
experiment, at the SAME N=1500, K=5 used by the privacy Pareto.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dtfl.controller.tifl import TiFLSelector
from dtfl.latent.latency import draw_completion_times
from dtfl.protocol.dropout import apply_dropout
from dtfl.protocol.release import decide_release
from dtfl.protocol.threshold import reconstruction_threshold
from dtfl.protocol.tiering import assign_tiers, tier_rosters
from dtfl.rng import RngHub
from dtfl.sim.round import RoundConfig
from dtfl.types import RoundDeadlines

import config as CFG
import run_utility_comparison as U


def count_exclusions(eng, m_min, tiers, tier_selector, rounds, seed):
    """Replay the participation/tiering/dropout/release path (no training) and
    tally, each round, how many available clients are excluded and why."""
    lcfg = eng._lcfg
    mu = eng._mu
    avail = eng._availability
    N = len(mu)
    controller = eng.adaptive_policy(tuple(k / tiers for k in range(1, tiers)))
    cutoffs = tuple(controller._cutoffs)
    deadlines = RoundDeadlines(round_index=0, cutoffs=cutoffs)
    rc = RoundConfig(m_min=m_min)

    # Local counts feed for the adaptive controller (mirrors federated_train's).
    class _Feed:
        def __init__(self): self._by = {}
        def record(self, r, counts): self._by[r] = counts
        def tier_counts(self, r): return self._by.get(r, {})
        def __len__(self): return len(self._by)
    feed = _Feed()
    avail_rng = np.random.default_rng(seed + 1)

    tot = {"available": 0, "selected_out": 0, "suppressed": 0, "dropout": 0, "contributed": 0}
    per_round = []
    for r in range(rounds):
        cutoffs_r = controller.next_deadlines(r, feed)
        deadlines = RoundDeadlines(round_index=r, cutoffs=tuple(cutoffs_r))
        available_mask = avail_rng.random(N) < avail
        avail_ids = np.flatnonzero(available_mask)
        n_avail = int(avail_ids.size)

        tau = draw_completion_times(mu[avail_ids], lcfg, np.random.default_rng(seed + 100 + r))
        tiers_local = assign_tiers(tau, deadlines)
        rosters = tier_rosters(tiers_local, tiers)
        feed.record(r, {k: int(rosters[k].size) for k in range(tiers)})

        if tier_selector is not None:
            selected = set(tier_selector(r, tiers, np.random.default_rng(seed + 7 + r)))
        else:
            selected = set(range(tiers))

        row = {"round": r, "available": n_avail, "selected_out": 0,
               "suppressed": 0, "dropout": 0, "contributed": 0}
        for k in range(tiers):
            sz = int(rosters[k].size)
            if sz == 0:
                continue
            if k not in selected:
                row["selected_out"] += sz
                continue
            outcome = apply_dropout(sz, rc.dropout_rates, np.random.default_rng(seed + 200 + r * tiers + k))
            n_k = outcome.active_count
            t_k = reconstruction_threshold(n_k, rc.m_min, rc.dropout_rates)
            decision = decide_release(outcome, rc.m_min, t_k)
            if not decision.released:
                row["suppressed"] += sz          # whole tier's clients excluded by the floor
                continue
            row["dropout"] += (sz - n_k)          # attrition within a released tier
            row["contributed"] += n_k
        per_round.append(row)
        for key in tot:
            tot[key] += row[key]
    return tot, per_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=CFG.PRIVACY_ROUNDS if hasattr(CFG, "PRIVACY_ROUNDS") else 80)
    ap.add_argument("--devices", type=int, default=CFG.DEVICES)     # 1500, matches Pareto
    ap.add_argument("--tiers", type=int, default=CFG.TIERS)         # 5, matches Pareto
    ap.add_argument("--m_min", type=int, default=77)               # true cliff operating point
    ap.add_argument("--seed", type=int, default=CFG.SEED)
    args = ap.parse_args()

    hub = RngHub(seed=args.seed)
    eng = U.make_engine(args.devices, args.seed) if hasattr(U, "make_engine") else None
    if eng is None:
        # Build an engine directly the same way run_config does.
        from dtfl.sim.engine import Engine, EngineConfig
        from dtfl.latent.config import LatentConfig
        eng = Engine(
            EngineConfig(seed=args.seed, num_devices=args.devices, num_rounds=args.rounds,
                         flhetbench=CFG.population_config()),
            LatentConfig(),
        )

    print(f"exclusion accounting | N={args.devices} K={args.tiers} "
          f"m_min={args.m_min} rounds={args.rounds}\n")

    schemes = [
        ("ours", None),
        ("tifl", TiFLSelector(args.tiers, speed_bias=1.0, floor=0.10)),
    ]
    rows = []
    for name, sel in schemes:
        # ours uses m_min; tifl runs with no suppression (m_min=1) but drops tiers via selection
        mm = args.m_min if name == "ours" else 1
        tot, per_round = count_exclusions(eng, mm, args.tiers, sel, args.rounds, args.seed)
        avail = tot["available"]
        excl = tot["selected_out"] + tot["suppressed"] + tot["dropout"]
        print(f"=== {name} (m_min={mm}) ===")
        print(f"  available (rostered)         : {avail}")
        print(f"  excluded: not-trained tier   : {tot['selected_out']}")
        print(f"  excluded: m_min suppression  : {tot['suppressed']}")
        print(f"  excluded: secure-agg dropout : {tot['dropout']}")
        print(f"  -> total excluded            : {excl}  ({100*excl/avail:.1f}% of available)")
        print(f"  -> contributed               : {tot['contributed']}  ({100*tot['contributed']/avail:.1f}%)\n")
        rows.append({"scheme": name, "m_min": mm, **tot,
                     "excluded_total": excl,
                     "excluded_pct": round(100 * excl / avail, 2)})

    out = Path("artifacts/results") / "exclusion_accounting.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "m_min", "available", "selected_out",
                                          "suppressed", "dropout", "contributed",
                                          "excluded_total", "excluded_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()