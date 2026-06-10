"""Configuration for the latent generative model.

All parameters of the latency-to-capability model live here in one typed place.
The single most important field is ``proxy_noise_eta`` -- the round-to-round
transient noise scale. Together with ``class_separation`` and
``within_class_spread`` it determines the signal-to-noise ratio

    SNR = class_separation^2 / (within_class_spread^2 + proxy_noise_eta^2)

which controls whether capability is readable from timing at all, and therefore
whether the attack (and the paper) has anything to work with. See
``02_Simulator_and_Latency_Model.md``.

Everything is on the LOG scale unless a field name says otherwise: latencies are
positive and right-skewed, so we model ``log tau`` as Gaussian-plus-tail and
exponentiate. ``class_means_log`` etc. are log-seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LatentConfig", "DriftConfig"]


@dataclass(frozen=True)
class DriftConfig:
    """Nonstationarity knobs. All default OFF for the clean-signal gate run.

    - ``regime_shift_prob``: per-round probability of a network-wide level shift
      (affects all classes equally; modeled as a shared additive term on log tau).
    - ``regime_shift_scale``: stddev (log) of that shared shift when it fires.
    - ``slow_drift_per_round``: stddev (log) of a slow random-walk applied to the
      global level each round.
    - ``device_walk_per_round``: stddev (log) of the per-device random walk on
      ``mu`` -- this is what bounds how long a device's "fingerprint" persists and
      makes the linkability-vs-horizon curve non-trivial.
    """

    regime_shift_prob: float = 0.0
    regime_shift_scale: float = 0.0
    slow_drift_per_round: float = 0.0
    device_walk_per_round: float = 0.0

    @property
    def enabled(self) -> bool:
        return any(
            v > 0.0
            for v in (
                self.regime_shift_prob,
                self.regime_shift_scale,
                self.slow_drift_per_round,
                self.device_walk_per_round,
            )
        )


@dataclass(frozen=True)
class LatentConfig:
    """Parameters of the latent capability + latency model.

    Defaults are the Step-1 starting point from the simulator spec; they are dials
    for calibration, not final values. ``proxy_noise_eta`` is the one we sweep.
    """

    # --- capability classes ---
    num_classes: int = 5  # C; class 1 = fastest/highest-capability
    # Population mixture over classes. Length must equal num_classes; sums to 1.
    # Default is non-uniform (middle-heavy) so tiers aren't trivially balanced.
    class_mixture: tuple[float, ...] = (0.10, 0.25, 0.30, 0.25, 0.10)

    # --- the signal/noise decomposition (all log-scale) ---
    # Class mean log-latency for class 1 (fastest). Higher classes are offset up
    # by multiples of ``class_separation``.
    base_mean_log: float = 0.0
    # Spacing between consecutive class means (log). The "signal". Larger => more
    # separable classes => easier attack.
    class_separation: float = 0.40  # Delta_class
    # Per-device persistent random effect stddev (log). The faint per-device
    # fingerprint that linkability exploits. 0 => same-class devices identical.
    within_class_spread: float = 0.15  # s_within
    # THE CENTRAL TUNABLE: transient, round-specific, capability-independent noise
    # stddev (log). The privacy-protecting blur. Swept 0.05 -> 0.8 in calibration.
    proxy_noise_eta: float = 0.20  # eta

    # --- heavy tail (straggler bursts / misclassification events) ---
    # With prob ``tail_prob`` add an Exponential(scale=tail_scale_log) spike to
    # log tau. Creates fast-device-looks-slow events: a privacy feature AND a
    # dropout source.
    tail_prob: float = 0.05
    tail_scale_log: float = 0.50

    # --- availability ---
    # Per-class participation probability per round. Faster classes available
    # more often (realistic: capable devices opt in more). Length == num_classes.
    class_availability: tuple[float, ...] = (0.30, 0.25, 0.20, 0.15, 0.10)

    # --- drift ---
    drift: DriftConfig = field(default_factory=DriftConfig)

    def __post_init__(self) -> None:
        if len(self.class_mixture) != self.num_classes:
            raise ValueError(
                f"class_mixture has {len(self.class_mixture)} entries, "
                f"expected num_classes={self.num_classes}"
            )
        if len(self.class_availability) != self.num_classes:
            raise ValueError(
                f"class_availability has {len(self.class_availability)} entries, "
                f"expected num_classes={self.num_classes}"
            )
        if abs(sum(self.class_mixture) - 1.0) > 1e-9:
            raise ValueError(f"class_mixture must sum to 1, got {sum(self.class_mixture)}")
        if self.num_classes < 2:
            raise ValueError("need at least 2 capability classes for a meaningful attack")
        for name, v in (
            ("class_separation", self.class_separation),
            ("within_class_spread", self.within_class_spread),
            ("proxy_noise_eta", self.proxy_noise_eta),
        ):
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")

    # ----- derived quantities -----

    def class_mean_log(self, capability_class: int) -> float:
        """Mean log-latency for a class (1-indexed). Class 1 is fastest."""
        if not 1 <= capability_class <= self.num_classes:
            raise ValueError(f"class {capability_class} out of range 1..{self.num_classes}")
        return self.base_mean_log + (capability_class - 1) * self.class_separation

    @property
    def signal_to_noise(self) -> float:
        """SNR = Delta_class^2 / (s_within^2 + eta^2). The gate-determining ratio.

        High => capability bleeds through every round (attack easy, privacy at
        risk). Low => capability barely observable (no defense story). Calibration
        targets the regime where the undefended attack has advantage AND defenses
        can move it.
        """
        denom = self.within_class_spread**2 + self.proxy_noise_eta**2
        if denom == 0:
            return float("inf")
        return (self.class_separation**2) / denom

    def with_eta(self, eta: float) -> LatentConfig:
        """Return a copy with a different proxy-noise scale (for sweeps)."""
        from dataclasses import replace

        return replace(self, proxy_noise_eta=eta)

    def mixture_array(self) -> np.ndarray:
        return np.asarray(self.class_mixture, dtype=np.float64)
