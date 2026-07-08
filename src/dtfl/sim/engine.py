"""The simulation engine: run R rounds over one population.

Responsibilities:
- build the latent population ONCE from a seed (so baselines share a population),
- per round: draw availability, step drift, run the round, append records,
- keep the per-round latent logs on the LATENT side (experimenter-only),
- expose only a TranscriptView for anything attacker-facing.

The engine is deliberately controller-agnostic: it takes a callable that
produces the deadline vector for each round from whatever it is allowed to see.
For Steps 0-2 we pass a fixed-deadline policy. Later, a controller from
dtfl.controller is plugged in here -- and note a real (non-oracle) controller may
only read the TranscriptView, never the latent logs.

Virtual clock: each round advances the clock by the round budget (the last
deadline), giving a wall-clock axis for time-to-accuracy without real timing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from dtfl.latent.config import LatentConfig
from dtfl.latent.drift import DriftState
from dtfl.latent.population import build_population
from dtfl.rng import RngHub
from dtfl.sim.round import RoundConfig, RoundLatentLog, run_round
from dtfl.transcript.bucket import Bucketer, single_bucket
from dtfl.transcript.store import TranscriptStore, TranscriptView
from dtfl.types import RoundDeadlines

__all__ = ["EngineConfig", "SimulationOutput", "Engine", "DeadlinePolicy"]

# A deadline policy maps (round_index, transcript_view) -> the round's cutoffs.
# Fixed policies ignore the view; adaptive controllers read it.
DeadlinePolicy = Callable[[int, TranscriptView], tuple[float, ...]]


@dataclass(frozen=True)
class EngineConfig:
    """Top-level run configuration.

    ``flhetbench`` selects the population source. When None (default), the
    parametric synthetic latent model is used (unchanged behaviour). When set to
    an ``FLHetBenchConfig``, devices are drawn from the FLHetBench real-world
    device database instead. Either way the engine receives the same
    ``LatentDeviceState`` list, so nothing downstream changes.
    """

    seed: int = 12345
    num_devices: int = 1000
    num_rounds: int = 100
    round_config: RoundConfig = field(default_factory=RoundConfig)
    flhetbench: "FLHetBenchConfig | None" = None


@dataclass
class SimulationOutput:
    """Everything a run produces.

    ``transcript`` is the attacker-facing store (hand its ``.view()`` to attacks).
    ``latent_logs`` and ``populations`` are experimenter-only ground truth used
    for diagnostics and for evaluating attack success against the truth.
    """

    transcript: TranscriptStore
    latent_logs: list[RoundLatentLog]
    # full-population latent labels (length N), for ground-truth attack evaluation
    true_classes: np.ndarray
    true_mu: np.ndarray
    # per-round merged updates and participation (for utility/merge diagnostics)
    merged_updates: list[np.ndarray | None]
    participation: list[int]
    virtual_time: list[float]  # cumulative wall-clock at end of each round


class Engine:
    """Runs the simulation. Holds latent state; exposes transcript view."""

    def __init__(self, config: EngineConfig, latent_config: LatentConfig):
        self._cfg = config
        self._lcfg = latent_config
        self._hub = RngHub(seed=config.seed)

        # Build the population once (shared across baselines via the seed).
        # Either the parametric synthetic model or the FLHetBench-grounded one;
        # both return the same LatentDeviceState list.
        if config.flhetbench is not None:
            from dtfl.latent.flhetbench import build_flhetbench_population

            pop = build_flhetbench_population(
                config.num_devices,
                config.flhetbench,
                self._hub.stream("latent.population"),
            )
        else:
            pop = build_population(
                config.num_devices, latent_config, self._hub.stream("latent.population")
            )
        self._mu = np.array([d.mu for d in pop], dtype=np.float64)
        self._classes = np.array([d.capability_class for d in pop], dtype=np.int64)
        self._availability = np.array([d.availability_rate for d in pop], dtype=np.float64)

        self._drift = DriftState(
            latent_config.drift,
            n_devices=config.num_devices,
            rng=self._hub.stream("latent.drift"),
        )

    def run(
        self,
        policy: DeadlinePolicy,
        bucketer: Bucketer = single_bucket,
    ) -> SimulationOutput:
        """Run all rounds under the given deadline policy and bucketer."""
        store = TranscriptStore()
        latent_logs: list[RoundLatentLog] = []
        merged_updates: list[np.ndarray | None] = []
        participation: list[int] = []
        virtual_time: list[float] = []
        clock = 0.0

        avail_rng = self._hub.stream("sim.availability")
        mu_current = self._mu  # drift may evolve this per round

        for r in range(self._cfg.num_rounds):
            # Deadlines for this round: policy reads only the transcript view.
            cutoffs = policy(r, store.view())
            deadlines = RoundDeadlines(round_index=r, cutoffs=tuple(cutoffs))

            # Availability draw for this round (per-device Bernoulli on its rate).
            available_mask = avail_rng.random(self._cfg.num_devices) < self._availability

            # Drift: advance global level and per-device walk.
            offset = self._drift.step()
            mu_current = self._drift.apply_device_walk(mu_current)

            # Independent per-round streams (named so adding components is stable).
            child = self._hub.child(f"round.{r}")
            result = run_round(
                round_index=r,
                deadlines=deadlines,
                mu=mu_current,
                classes=self._classes,
                available_mask=available_mask,
                config=self._lcfg,
                round_config=self._cfg.round_config,
                latency_rng=child.stream("latency"),
                protocol_rng=child.stream("protocol"),
                bucketer=bucketer,
                shared_log_offset=offset,
            )

            store.extend(result.records)
            latent_logs.append(result.latent_log)
            merged_updates.append(result.merged_update)
            participation.append(result.total_participation)

            clock += float(deadlines.cutoffs[-1])  # round budget = last deadline
            virtual_time.append(clock)

        return SimulationOutput(
            transcript=store,
            latent_logs=latent_logs,
            true_classes=self._classes.copy(),
            true_mu=self._mu.copy(),
            merged_updates=merged_updates,
            participation=participation,
            virtual_time=virtual_time,
        )

    # --- helpers for fixed-deadline policies (Steps 0-2) ---

    def calibrate_fixed_deadlines(
        self, quantiles: tuple[float, ...], probe_rounds: int = 5
    ) -> tuple[float, ...]:
        """Pick fixed deadlines at completion-time quantiles via a latent probe.

        Runs a few availability+latency draws (latent, not recorded) to estimate
        the completion-time distribution, then sets deadlines at the requested
        quantiles plus a covering final tier. This is an EXPERIMENTER calibration
        using ground truth -- a fixed policy derived this way is still
        transcript-blind at run time (it ignores the view).
        """
        probe_rng = self._hub.stream("calib.latency")
        avail_rng = self._hub.stream("calib.availability")
        from dtfl.latent.latency import draw_completion_times

        samples = []
        for _ in range(probe_rounds):
            mask = avail_rng.random(self._cfg.num_devices) < self._availability
            ids = np.flatnonzero(mask)
            samples.append(draw_completion_times(self._mu[ids], self._lcfg, probe_rng))
        all_tau = np.concatenate(samples)
        cuts = list(np.quantile(all_tau, quantiles))
        cuts.append(float(all_tau.max() * 1.5))  # covering final tier
        # Enforce strict increase (quantiles can tie on discrete-ish data).
        for i in range(1, len(cuts)):
            if cuts[i] <= cuts[i - 1]:
                cuts[i] = cuts[i - 1] + 1e-6
        return tuple(cuts)

    def adaptive_policy(
        self,
        quantiles: tuple[float, ...],
        *,
        eta0: float = 0.5,
        broadcast_scale: float | None = None,
    ):
        """Build a fresh quantile-tracking (Robbins-Monro) deadline policy.

        This is the DEPLOYABLE controller: it sets deadlines online from the
        aggregate transcript counts alone and NEVER probes the latent
        completion-time distribution (unlike ``calibrate_fixed_deadlines``). The
        server adapts deadlines to what it observes each round.

        ``quantiles`` are the same interior target fractions the fixed policy
        takes (e.g. k/K for k=1..K-1); this method appends the covering 1.0 target
        and derives K = len(quantiles) + 1 tiers, matching the fixed-policy tier
        count exactly. The controller is stateful, so a NEW instance is returned on
        every call -- pass its ``.policy()`` to ``run``.

        ``broadcast_scale`` is a PUBLIC, coarse expected-latency magnitude the
        server may announce (e.g. a nominal round-duration budget). It anchors the
        equal-width init cutoffs to roughly the right order of magnitude so the
        Robbins-Monro iteration converges in a few rounds rather than spending many
        rounds climbing to scale. It is a single public scalar, NOT per-device
        latent data -- the controller still observes nothing about individual
        devices. If None, a coarse magnitude is derived from the population's
        median base latency, rounded to one significant figure, which is a
        defensible stand-in for a scale the operator would know a priori.
        """
        from dtfl.controller.quantile import QuantileTrackingController

        targets = tuple(quantiles) + (1.0,)
        K = len(targets)

        if broadcast_scale is None:
            # Coarse public magnitude: median exp(mu), rounded to 1 sig fig. This
            # is an order-of-magnitude operator prior, not a distribution probe.
            import math

            med = float(np.median(np.exp(self._mu)))
            if med > 0:
                mag = 10 ** math.floor(math.log10(med))
                broadcast_scale = round(med / mag) * mag
            else:
                broadcast_scale = 1.0

        # Equal-width init cutoffs across (0, broadcast_scale], covering tier at ~2x.
        step = broadcast_scale / K
        init = tuple(float(step * (i + 1)) for i in range(K - 1)) + (
            float(broadcast_scale * 2.0),
        )
        return QuantileTrackingController(
            K, targets, init, eta0=eta0, hi=float(broadcast_scale * 5.0)
        )