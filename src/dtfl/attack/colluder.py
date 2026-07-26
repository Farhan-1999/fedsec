"""L1 colluding-client adversary (server + a corrupted fraction of clients).

THREAT MODEL
------------
A single coherent adversary: the honest-but-curious SERVER, colluding with a
fraction ``x`` of the client population. This replaces the earlier co-located
observer as the primary (L1) adversary because it requires no physical access,
no side channel and no broken cryptography -- only that the adversary has
registered or corrupted some clients, which is cheap in cross-device FL.

What the adversary legitimately holds:

  1. The full transcript: per released tier-round ``(k, n_k, b_k, S_k)`` -- the
     participant count, the release bucket, and the masked aggregate. This is
     everything the server sees under perfect SecAgg.
  2. The broadcast deadline vector each round (the server sets it).
  3. For every COLLUDING client: its own tier each round (colluders self-assign,
     so they know their own tier by construction) and its own capability class.

It does NOT hold any honest device's tier membership, latent class, completion
time, or individual update.

THE LINKAGE ROUTE
-----------------
Under SecAgg with rotating pseudonyms the transcript alone contains no honest
device identifier, so a transcript-only server cannot attribute a tier to an
honest device. The colluders supply the missing route, in two steps.

Step 1 -- subtract the known. Because the adversary knows exactly which of its
own clients sat in tier ``k`` at round ``r``, it removes them from the published
count:

    honest_{k,r} = n_{k,r} - c_{k,r}

This is the size of the HONEST anonymity set in that released tier-round, and
its members are known to be drawn from the population minus the colluders.

Step 2 -- intersect across rounds. A device's visible SIGNATURE over a horizon
is the sequence of ``(tier, release_bucket)`` atoms it produced in released
tier-rounds (absence is itself an atom -- the adversary notices a device going
quiet). Devices sharing a signature are mutually indistinguishable, so the
adversary's uncertainty about a device is the size of its signature equivalence
class ``|E_i(h)|``. Each additional round can only split classes, never merge
them, so the crowd a device hides in shrinks monotonically. The colluders
accelerate this: every colluder inside a class is a member the adversary can
already name and remove, leaving the honest residue

    honest_class_i = |E_i(h)| - (colluders in E_i(h))

RESOLUTION IS DERIVED, NOT ASSUMED
----------------------------------
Whether the adversary can attribute an observation to a specific honest device
is therefore not a free parameter: it is determined by that device's honest
equivalence-class size. A device alone in its class after colluder removal is
fully resolved; a device sharing its class with ``q`` honest others is resolved
only with probability ``1/q``, because the adversary cannot do better than guess
uniformly among indistinguishable candidates. We use exactly that:

    P(attribute observation of device i) = 1 / honest_class_i

with ``honest_class_i`` computed from the realized signatures. No resolution knob
is introduced; every quantity is either published in the transcript or owned by
the colluders.

THE LINKING WINDOW
------------------
Signatures are evaluated over a sliding window of ``W`` rounds rather than the
whole run. This matters technically and is also the more realistic reading. Over
a full horizon of ``R`` rounds the space of atom sequences vastly exceeds the
population, so essentially every device is unique, ``honest_class_i = 1``, and
the colluder subtraction removes nobody -- the adversary degenerates onto the
plain released-round observer. A bounded window instead models an adversary that
links a target over a recent history: it may join mid-training, devices churn,
and stale atoms stop being informative. ``W`` is therefore an explicit parameter
of the threat model rather than an implicit assumption of perfect recall, and
the leak should be reported as a function of it.

WHY m_min IS THE RIGHT DEFENSE HERE
------------------------------------
Two effects, both structural:

  * Suppression deletes atoms. A suppressed tier publishes no count, so there is
    no ``n_k`` to subtract colluders from and no atom enters any signature --
    the intersection has less to work with.
  * The floor inflates the crowd. The honest anonymity set in a released tier is
    ``n_k - c_k``, not ``n_k``. With collusion rate ``x`` we have
    ``E[c_k] = x n_k``, so guaranteeing an honest anonymity set of at least ``a``
    requires

        m_min >= a / (1 - x)

    i.e. the floor must be INFLATED by the collusion rate. At ``x = 0`` this
    recovers the transcript-only bound ``1/m_min``; at ``x = 0.5`` the floor must
    be doubled for the same guarantee.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

__all__ = ["ColluderConfig", "build_colluder_observations"]


class ColluderConfig:
    """Parameters of the colluding-client adversary.

    Parameters
    ----------
    collusion_rate:
        Fraction ``x`` of the client population corrupted by / registered to the
        adversary. Colluders participate in training and self-assign to tiers
        exactly like honest clients; the adversary knows their tier and class.
    window:
        Number of recent rounds over which the adversary links a target (the
        sliding signature window ``W``). A device's crowd at round ``r`` is the
        set of honest devices sharing its atom sequence over rounds
        ``(r-W, r]``. Small ``W`` = a weak linker with short memory and large
        crowds; large ``W`` = a strong linker that eventually isolates everyone.
        Full-horizon linking (``W >= R``) makes almost every device unique and
        degenerates this adversary onto the plain released-round observer, which
        is why the window is bounded and stated explicitly.
    seed:
        RNG seed for choosing colluders and for resolving attributions.

    Note there is deliberately no "resolution" parameter: how well the adversary
    can single out an honest device is derived from its honest windowed
    equivalence-class size (see module docstring).
    """

    def __init__(
        self,
        collusion_rate: float = 0.10,
        window: int = 8,
        seed: int = 0,
    ):
        if not 0.0 <= collusion_rate < 1.0:
            raise ValueError("collusion_rate must be in [0, 1)")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.collusion_rate = float(collusion_rate)
        self.window = int(window)
        self.seed = int(seed)

    def required_floor(self, target_anonymity: float) -> float:
        """m_min needed for an honest anonymity set of ``target_anonymity``.

        Implements ``m_min >= a / (1 - x)`` (see module docstring).
        """
        return float(target_anonymity) / (1.0 - self.collusion_rate)


def build_colluder_observations(out, cfg: ColluderConfig):
    """Build the adversary's per-device observations under the colluder model.

    Returns
    -------
    records:
        ``{device_id: [(round, tier), ...]}`` for devices the adversary can
        attribute. Colluders are fully attributable (it owns them); an honest
        device's observations are attributable with probability
        ``1 / honest_equivalence_class_size``, derived from the realized
        signatures rather than assumed.
    colluder_ids:
        The device ids the adversary controls.
    stats:
        Diagnostics: mean honest anonymity set per released tier-round, mean
        honest equivalence-class size, honest attribution rate.
    """
    view = out.transcript.view()
    released = {(r.round_index, r.tier_index) for r in view.released()}
    counts = {(r.round_index, r.tier_index): r.count for r in view.released()}
    buckets = {
        (r.round_index, r.tier_index): r.release_bucket for r in view.released()
    }

    # --- choose the colluding clients -----------------------------------
    n_devices = int(out.true_classes.shape[0])
    rng = np.random.default_rng(cfg.seed)
    n_coll = int(round(cfg.collusion_rate * n_devices))
    colluder_ids = (
        rng.choice(n_devices, size=n_coll, replace=False)
        if n_coll > 0
        else np.empty(0, dtype=int)
    )
    colluder_set = {int(i) for i in colluder_ids}

    empty_stats = {
        "collusion_rate": cfg.collusion_rate,
        "n_colluders": int(n_coll),
        "mean_honest_set": 0.0,
        "mean_honest_class": 0.0,
        "honest_attribution_rate": 0.0,
        "n_attributed": 0,
    }

    # --- pass 1: per-device released atoms, and honest set sizes ---------
    per_round_atom: dict[int, dict[int, tuple[int, int]]] = {}
    honest_sets: list[float] = []
    horizon = 0
    for log in out.latent_logs:
        r = log.round_index
        horizon = max(horizon, r + 1)
        for k, ids in enumerate(log.active_device_ids):
            if (r, k) not in released:
                continue  # suppressed: no count published, nothing to subtract
            ids_int = [int(d) for d in ids]
            c_k = sum(1 for d in ids_int if d in colluder_set)
            n_k = counts.get((r, k), len(ids_int))
            honest_sets.append(max(1.0, float(n_k) - float(c_k)))
            atom = (k, buckets.get((r, k), -1))
            for d in ids_int:
                per_round_atom.setdefault(d, {})[r] = atom

    if not per_round_atom:
        return {}, colluder_ids, empty_stats

    # --- pass 2: WINDOWED signature equivalence classes ------------------
    # A full-horizon signature (all R rounds) is far too fine-grained: the space
    # of length-R atom tuples vastly exceeds the population, so essentially every
    # device is unique, |E_i| = 1, and the colluder subtraction removes nobody.
    # Empirically this made attribution probability 1 everywhere and collapsed
    # this adversary onto the plain released-round observer.
    #
    # We therefore evaluate signatures over a SLIDING WINDOW of ``window`` rounds.
    # This is both the technically meaningful choice and the more realistic one:
    # a colluding adversary links a target over a bounded recent history (it may
    # join mid-training, devices churn, and stale atoms stop being informative),
    # rather than over the entire training run. The window is an explicit,
    # documented parameter of the threat model instead of an implicit assumption
    # that the adversary has perfect recall of every round.
    #
    # Attribution is then decided PER ROUND from the local crowd: at round r the
    # adversary's uncertainty about device i is the number of honest devices
    # sharing i's windowed signature ending at r, and it can do no better than
    # guess uniformly among them.
    W = max(1, int(cfg.window))
    rounds_sorted = sorted({r for atoms in per_round_atom.values() for r in atoms})

    # window_sig[d][r] = tuple of atoms for device d over rounds (r-W, r]
    window_sig: dict[int, dict[int, tuple]] = {d: {} for d in per_round_atom}
    for d, atoms in per_round_atom.items():
        for r in atoms:
            lo = max(0, r - W + 1)
            window_sig[d][r] = tuple(
                atoms.get(rr, (-1, -1)) for rr in range(lo, r + 1)
            )

    # per round, count how many HONEST devices carry each windowed signature
    honest_counts_by_round: dict[int, Counter] = {r: Counter() for r in rounds_sorted}
    for d, sigs in window_sig.items():
        if d in colluder_set:
            continue  # colluders are already named; they do not hide anyone
        for r, sig in sigs.items():
            honest_counts_by_round[r][sig] += 1

    # --- pass 3: attribute observations ---------------------------------
    records: dict[int, list[tuple[int, int]]] = {}
    n_attrib = 0
    n_honest_obs = 0
    n_honest_attrib = 0
    honest_classes: list[float] = []

    for d, atoms in per_round_atom.items():
        if d in colluder_set:
            # the adversary owns this client: tier known exactly, every round
            for r, (k, _b) in sorted(atoms.items()):
                records.setdefault(d, []).append((r, k))
                n_attrib += 1
            continue
        # honest device: resolvable only within its honest windowed class,
        # decided independently at each round from the local crowd.
        for r, (k, _b) in sorted(atoms.items()):
            sig = window_sig[d][r]
            q = max(1, honest_counts_by_round[r].get(sig, 1))
            honest_classes.append(float(q))
            n_honest_obs += 1
            if rng.random() < 1.0 / q:
                records.setdefault(d, []).append((r, k))
                n_attrib += 1
                n_honest_attrib += 1

    stats = {
        "collusion_rate": cfg.collusion_rate,
        "n_colluders": int(n_coll),
        "mean_honest_set": float(np.mean(honest_sets)) if honest_sets else 0.0,
        "mean_honest_class": float(np.mean(honest_classes)) if honest_classes else 0.0,
        "honest_attribution_rate": (
            float(n_honest_attrib) / n_honest_obs if n_honest_obs > 0 else 0.0
        ),
        "n_attributed": int(n_attrib),
    }
    return records, colluder_ids, stats