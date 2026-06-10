"""Per-device latent state.

A device's persistent ground truth is its capability class ``theta_i`` and its
base log-latency ``mu_i = m_c + u_i`` (class mean plus a persistent per-device
random effect). ``mu_i`` is the fingerprint the linkability attack would exploit
if it could see it -- it cannot; only the tier/timing the device lands in is ever
observable, and that is filtered through the round-specific proxy noise.

These objects live exclusively in the latent layer. They are ``LatentDeviceState``
instances (the type the attack package can never receive).
"""

from __future__ import annotations

import numpy as np

from dtfl.latent.config import LatentConfig
from dtfl.types import LatentDeviceState

__all__ = ["make_device"]


def make_device(
    device_id: int,
    capability_class: int,
    config: LatentConfig,
    rng: np.random.Generator,
) -> LatentDeviceState:
    """Construct one device's latent state given its (already drawn) class.

    The per-device random effect ``u_i ~ Normal(0, s_within)`` is drawn here and
    folded into ``mu``. Availability rate is the class-level rate (kept per-device
    so future variants can add within-class availability heterogeneity without
    changing the type).
    """
    class_mean = config.class_mean_log(capability_class)
    u_i = rng.normal(0.0, config.within_class_spread)
    mu = class_mean + u_i
    availability_rate = config.class_availability[capability_class - 1]
    return LatentDeviceState(
        device_id=device_id,
        capability_class=capability_class,
        mu=mu,
        availability_rate=availability_rate,
    )
