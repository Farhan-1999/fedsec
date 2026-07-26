"""FeDSC-style adaptive UE clustering (Liu et al., IEEE T-COMM 72(12), 2024).

Baseline for the utility and exclusion-accounting comparisons. FeDSC is the
modern capability-aware clustering counterpart to TiFL: instead of fixed
latency tiers, it re-clusters devices ONLINE with a modified, training-time
BIRCH algorithm, then aggregates within clusters.

WHAT THIS IMPLEMENTS (faithful to Algorithm 1 of the paper)
-----------------------------------------------------------
* Clustering feature (Definition 1). For cluster u over devices with training
  times X_i, the CF triple is ``CF = (n_u, LS, SS)`` with ``LS = sum_i X_i`` and
  ``SS = sum_i X_i^2``. From the CF we derive, without revisiting members:
    - the cluster head / centroid  e0_u = LS / n_u,
    - the maximal intra-cluster training-time difference T^Delta_u, computed via
      the CF identity  mean_{i,j}(X_i - X_j)^2 = 2 * n_u * (SS - LS^2/n_u)
                                                / (n_u * (n_u - 1)).
* Two branching parameters (Section IV-B): ``t_bar`` bounds T^Delta_u, and
  ``beta`` bounds how many devices a cluster may hold (0 <= beta <= N-1).
* Step I, initial clustering (Alg. 1, lines 2-22). Devices are sorted by
  computing time and inserted sequentially. A device joins an existing cluster
  only if that cluster would still satisfy ``T^Delta_u < t_bar`` and the size
  bound; among admissible clusters it takes the nearest centroid
  (``u* = argmin ||X_n - e0_u||^2``). Otherwise it opens a new cluster.
* Step II, dynamic CH re-selection (Alg. 1, lines 21-24): after an insertion the
  cluster's centroid is recomputed, so the head tracks the cluster's evolution.
* Periodic re-clustering: the paper re-runs Algorithm 1 "at the beginning of
  every c cluster iterations"; ``recluster_every`` is that constant c.

WHAT THIS DELIBERATELY DOES NOT IMPLEMENT (and why)
---------------------------------------------------
* ASYNCHRONOUS GLOBAL AGGREGATION (paper Eq. 12,
  ``w_T = (1 - a_t) w_{T-1} + a_t w_{u,r}``, the server updating on receipt of
  ANY single cluster model). Our harness is synchronous and every method under
  comparison shares one FedAvg-equivalent merge, which is what makes the
  utility comparison controlled -- an accuracy difference is then attributable
  to WHICH devices participate, not to a different optimizer. Substituting a
  staleness-weighted EMA for this baseline alone would confound exactly that.
  The baseline is therefore reported as "FeDSC-sync": FeDSC's clustering under
  our shared synchronous merge. Its clustering is faithful; its global
  aggregation is not the paper's.
* The wireless-channel model (SINR, path loss, bandwidth allocation) behind the
  transmission-time term ``T^{n,B}``. Our population supplies measured
  end-to-end completion times (FLHetBench), so ``X_i`` is the realized training
  time directly rather than a synthesized compute-plus-transmit decomposition.
  Inventing a channel model would mean tuning a baseline's physics.

PRIVACY NOTE (this is the point of the baseline)
------------------------------------------------
FeDSC clusters on COLLECTED PER-DEVICE PROFILES. Algorithm 1 takes as input the
per-device computing capability {C_t^1..C_t^N}, UE location, transmit power and
fading parameters, and line 3 sorts all devices by their computing time -- an
operation only a party holding every device's value can perform. This class
mirrors that: ``next_deadlines_from_times`` receives the per-device time vector.
That is precisely the disclosure our framework avoids, so this baseline is the
capability-aware reference point for the comparison, not a privacy-preserving
scheme. It is a BASELINE and is not part of our framework.

INTEGRATION
-----------
``federated_train`` partitions devices with ``assign_tiers(tau, deadlines)``, so
a clustering is expressed to the engine as a cutoff vector. After running BIRCH
we place cutoffs at the midpoints between adjacent cluster centroids, which
reproduces the cluster partition (clusters are intervals in training time, since
insertion is nearest-centroid over a scalar). When BIRCH yields more clusters
than the configured budget ``num_tiers``, the largest ``num_tiers`` clusters are
kept by mass; when it yields fewer, the remaining cutoffs are spread above the
top centroid so the vector keeps the strictly-increasing length the engine
requires.
"""
from __future__ import annotations

import numpy as np

from dtfl.controller.base import project_monotone

__all__ = ["FeDSCClusterer", "FeDSCController"]


class _CF:
    """Clustering feature triple (n, LS, SS) for scalar training times."""

    __slots__ = ("n", "ls", "ss")

    def __init__(self, x: float):
        self.n = 1
        self.ls = float(x)
        self.ss = float(x) * float(x)

    @property
    def centroid(self) -> float:
        return self.ls / self.n

    def delta_if_added(self, x: float) -> float:
        """T^Delta_u the cluster would have if ``x`` were inserted.

        Uses the CF identity so no member list is needed:
            sum_{i,j} (X_i - X_j)^2 = 2 n (SS - LS^2 / n)
        and T^Delta = sqrt( that / (n (n-1)) ).
        """
        n = self.n + 1
        if n < 2:
            return 0.0
        ls = self.ls + x
        ss = self.ss + x * x
        num = 2.0 * n * (ss - (ls * ls) / n)
        den = n * (n - 1)
        return float(np.sqrt(max(0.0, num / den)))

    def add(self, x: float) -> None:
        self.n += 1
        self.ls += x
        self.ss += x * x


class FeDSCClusterer:
    """Modified training-time BIRCH clustering of Algorithm 1.

    Parameters
    ----------
    t_bar:
        Threshold on the maximal intra-cluster training-time difference
        (``t_bar`` in the paper). Smaller -> more, tighter clusters.
    beta:
        Branching factor: maximum devices per cluster. Paper sets
        ``0 <= beta <= N - 1`` (they use beta = N - 1).
    """

    def __init__(self, t_bar: float, beta: int):
        self.t_bar = float(t_bar)
        self.beta = int(beta)

    def fit(self, times: np.ndarray) -> list[float]:
        """Cluster ``times``; return cluster centroids in increasing order.

        Implements Step I (sequential insertion, admissibility test, nearest
        admissible centroid) and Step II (centroid update after insertion).
        """
        x = np.asarray(times, dtype=np.float64).ravel()
        if x.size == 0:
            return []
        order = np.argsort(x, kind="mergesort")  # Alg. 1 line 3: sort by time
        clusters: list[_CF] = []
        for idx in order:
            xi = float(x[idx])
            # admissible clusters: intra-cluster spread and size bound both hold
            admissible = [
                (abs(xi - c.centroid), j)
                for j, c in enumerate(clusters)
                if c.delta_if_added(xi) < self.t_bar and c.n < self.beta
            ]
            if admissible:
                # Alg. 1 line 20: u* = argmin ||X_n - e0_u||^2
                _, j = min(admissible)
                clusters[j].add(xi)  # Step II: centroid updates implicitly
            else:
                clusters.append(_CF(xi))  # Alg. 1 lines 16-17: new cluster, UE is CH
        return sorted(c.centroid for c in clusters)

    def fit_sizes(self, times: np.ndarray) -> list[tuple[float, int]]:
        """Like :meth:`fit` but returns ``(centroid, size)`` pairs."""
        x = np.asarray(times, dtype=np.float64).ravel()
        if x.size == 0:
            return []
        order = np.argsort(x, kind="mergesort")
        clusters: list[_CF] = []
        for idx in order:
            xi = float(x[idx])
            admissible = [
                (abs(xi - c.centroid), j)
                for j, c in enumerate(clusters)
                if c.delta_if_added(xi) < self.t_bar and c.n < self.beta
            ]
            if admissible:
                _, j = min(admissible)
                clusters[j].add(xi)
            else:
                clusters.append(_CF(xi))
        return sorted(((c.centroid, c.n) for c in clusters), key=lambda p: p[0])


class FeDSCController:
    """Deadline controller whose cutoffs come from FeDSC's adaptive clustering.

    Unlike our framework's controller -- which reads only aggregate per-tier
    counts -- this one is handed the per-device training times each round, which
    is exactly the capability profile FeDSC's server collects. It is a baseline
    representing the capability-aware family, not a deployable private scheme.

    Parameters
    ----------
    num_tiers:
        Cutoff budget the engine expects (``K``). BIRCH's natural cluster count
        is reconciled to this; see module docstring.
    t_bar_quantile:
        ``t_bar`` is set as this quantile of pairwise-time spread on the first
        round's observed times, so the threshold adapts to the population's
        actual time scale instead of hard-coding the paper's 3 ms (their
        wireless setting, not ours).
    recluster_every:
        The paper's constant ``c``: re-run Algorithm 1 every ``c`` rounds.
    """

    def __init__(
        self,
        num_tiers: int,
        t_bar_quantile: float = 0.30,
        recluster_every: int = 5,
        beta: int | None = None,
        min_spacing: float = 1e-3,
    ):
        self.num_tiers = int(num_tiers)
        self.t_bar_quantile = float(t_bar_quantile)
        self.recluster_every = max(1, int(recluster_every))
        self.beta = beta
        self.min_spacing = float(min_spacing)
        self._cutoffs: tuple[float, ...] | None = None
        self._t_bar: float | None = None
        self._last_cluster_count: int | None = None

    # -- internal ---------------------------------------------------------
    def _calibrate_t_bar(self, times: np.ndarray) -> float:
        """Choose t_bar so BIRCH's natural cluster count is near the K budget.

        The paper fixes ``t_bar = 3 ms`` for their wireless setting; that
        constant is meaningless on our measured-latency scale, and an arbitrary
        choice would hand the baseline an arbitrary granularity. We instead
        bisect on t_bar until Algorithm 1 returns approximately ``num_tiers``
        clusters, which is the fairest reading of the paper's intent (their
        Section VI likewise picks the comparator's cluster count from the
        observed spread of update times, via
        ``K = ceil(mean_r (max_n T - min_n T) / t_bar)``).

        Calibrating to the budget also keeps the comparison matched: FeDSC and
        our framework then partition the same population into the same number of
        groups, so a utility difference reflects WHERE the boundaries fall, not
        how many there are.
        """
        x = np.asarray(times, dtype=np.float64).ravel()
        if x.size < 2:
            return 1.0
        spread = float(np.quantile(x, 0.95) - np.quantile(x, 0.05))
        spread = max(spread, 1e-6)
        beta = self.beta if self.beta is not None else max(1, x.size - 1)
        target = self.num_tiers

        def count(tb: float) -> int:
            return len(FeDSCClusterer(t_bar=tb, beta=beta).fit(x))

        # Bisect on t_bar: larger t_bar -> fewer clusters (monotone).
        lo, hi = 1e-4 * spread, 4.0 * spread
        if count(hi) > target:
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if count(mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6 * spread:
                break
        return max(1e-6, 0.5 * (lo + hi))

    def _cutoffs_from_clusters(
        self, pairs: list[tuple[float, int]], lo: float, hi: float
    ) -> tuple[float, ...]:
        """Convert (centroid, size) clusters to the engine's K-1 cutoffs.

        Clusters are intervals in training time (insertion is nearest-centroid
        over a scalar), so the boundary between adjacent clusters is the
        midpoint of their centroids. When BIRCH returns more clusters than the
        budget allows, we keep the boundaries that split the population most
        evenly by MASS rather than by index -- picking every j-th boundary would
        merge all the heavily-populated fast clusters into one tier and waste
        the remaining tiers on the sparse tail.
        """
        need = self.num_tiers - 1
        if need <= 0:
            return ()
        cs = [c for c, _ in pairs]
        ns = [n for _, n in pairs]
        if len(cs) >= 2:
            mids = [0.5 * (cs[i] + cs[i + 1]) for i in range(len(cs) - 1)]
            # cumulative population mass strictly below each boundary
            cum = np.cumsum(ns[:-1], dtype=np.float64)
            total = float(sum(ns))
            mass = cum / max(1.0, total)
        else:
            mids, mass = [], np.array([])

        if len(mids) > need:
            # target equal population mass per retained tier
            targets = np.linspace(0.0, 1.0, need + 2)[1:-1]
            chosen, used = [], set()
            for tgt in targets:
                j = int(np.argmin(np.abs(mass - tgt)))
                while j in used and len(used) < len(mids):
                    j = (j + 1) % len(mids)
                used.add(j)
                chosen.append(mids[j])
            mids = sorted(chosen)
        elif len(mids) < need:
            # Fewer clusters than the budget: pad above the top boundary so the
            # vector stays strictly increasing (extra tiers simply stay empty).
            top = mids[-1] if mids else (float(np.mean(cs)) if cs else lo)
            step = max(self.min_spacing, 0.05 * max(1e-6, hi - lo))
            while len(mids) < need:
                top = top + step
                mids.append(top)
        return project_monotone(np.array(mids, dtype=np.float64), self.min_spacing, lo, hi)

    # -- public API -------------------------------------------------------
    def next_deadlines_from_times(
        self, round_index: int, times: np.ndarray
    ) -> tuple[float, ...]:
        """Return this round's cutoffs, re-clustering every ``c`` rounds.

        ``times`` is the per-device training-time vector the FeDSC server
        collects -- the capability disclosure this baseline exemplifies.
        """
        x = np.asarray(times, dtype=np.float64).ravel()
        recluster = (
            self._cutoffs is None or round_index % self.recluster_every == 0
        )
        if not recluster:
            return self._cutoffs  # type: ignore[return-value]
        if x.size == 0:
            if self._cutoffs is not None:
                return self._cutoffs
            return tuple(
                float(i + 1) for i in range(max(0, self.num_tiers - 1))
            )
        if self._t_bar is None:
            self._t_bar = self._calibrate_t_bar(x)
        beta = self.beta if self.beta is not None else max(1, x.size - 1)
        clusterer = FeDSCClusterer(t_bar=self._t_bar, beta=beta)
        pairs = clusterer.fit_sizes(x)
        self._last_cluster_count = len(pairs)
        lo = float(x.min())
        hi = float(x.max())
        if hi <= lo:
            hi = lo + max(self.min_spacing, 1.0)
        self._cutoffs = self._cutoffs_from_clusters(pairs, lo, hi)
        return self._cutoffs

    @property
    def last_cluster_count(self) -> int | None:
        """Number of clusters BIRCH produced at the last re-clustering."""
        return self._last_cluster_count