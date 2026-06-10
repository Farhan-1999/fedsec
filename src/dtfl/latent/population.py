"""Population construction and latent-only diagnostics.

``build_population`` draws N devices' classes from the mixture and constructs
their latent states. ``class_tier_mutual_information`` is a GROUND-TRUTH
diagnostic: it measures how much a device's capability class is revealed by the
tier it lands in, using latent labels the attacker never sees. It lives in the
latent layer on purpose -- the import boundary keeps it unreachable from
``attack``. We use it to (a) verify the eta knob behaves (SNR-monotonicity test)
and (b) calibrate eta to a defensible regime before trusting any attack number.
"""

from __future__ import annotations

import numpy as np

from dtfl.latent.config import LatentConfig
from dtfl.latent.device import make_device
from dtfl.types import LatentDeviceState

__all__ = [
    "build_population",
    "draw_classes",
    "class_tier_mutual_information",
]


def draw_classes(n: int, config: LatentConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw N capability classes (1-indexed) from the population mixture."""
    # rng.choice over 1..C with mixture probabilities.
    classes = rng.choice(
        np.arange(1, config.num_classes + 1),
        size=n,
        p=config.mixture_array(),
    )
    return classes.astype(np.int64)


def build_population(
    n: int,
    config: LatentConfig,
    rng: np.random.Generator,
) -> list[LatentDeviceState]:
    """Construct N latent device states.

    Classes are drawn from the mixture; each device gets its own persistent
    random effect (inside ``make_device``). The returned list is index-aligned
    with device_id 0..n-1.
    """
    classes = draw_classes(n, config, rng)
    return [
        make_device(device_id=i, capability_class=int(classes[i]), config=config, rng=rng)
        for i in range(n)
    ]


def class_tier_mutual_information(
    classes: np.ndarray,
    tiers: np.ndarray,
) -> float:
    """Mutual information I(class; tier) in nats. GROUND-TRUTH diagnostic only.

    Measures how much the assigned tier reveals about the true capability class.
    This is the quantity the proxy-noise knob is supposed to control: raising eta
    should drive this DOWN smoothly. It is computed from latent labels and must
    never be exposed to the attacker.

    Parameters
    ----------
    classes:
        True capability classes for a set of participations, shape (m,).
    tiers:
        Assigned tier index for the same participations, shape (m,). Use a
        sentinel (e.g. -1) for missed-deadline dropouts; they form their own bin.

    Returns
    -------
    Mutual information in nats (>= 0). 0 means tier reveals nothing about class.
    """
    classes = np.asarray(classes).ravel()
    tiers = np.asarray(tiers).ravel()
    if classes.shape != tiers.shape:
        raise ValueError("classes and tiers must have the same shape")
    m = classes.shape[0]
    if m == 0:
        return 0.0

    # Joint histogram over (class, tier).
    cls_vals, cls_idx = np.unique(classes, return_inverse=True)
    tier_vals, tier_idx = np.unique(tiers, return_inverse=True)
    joint = np.zeros((cls_vals.size, tier_vals.size), dtype=np.float64)
    np.add.at(joint, (cls_idx, tier_idx), 1.0)
    joint /= m

    p_cls = joint.sum(axis=1, keepdims=True)
    p_tier = joint.sum(axis=0, keepdims=True)
    outer = p_cls @ p_tier

    # MI = sum p(x,y) log( p(x,y) / (p(x)p(y)) ), summing only over nonzero cells.
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / outer[nz])))
    # Numerical floor: MI is nonnegative; clamp tiny negatives from roundoff.
    return max(mi, 0.0)
