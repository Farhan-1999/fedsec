"""dtfl experiments package.

Run scripts and plot scripts for the privacy/utility evaluation. These live
OUTSIDE the dtfl package because they legitimately read latent logs to (1)
reconstruct the per-device observed tier sequences an attacker watching those
devices would obtain, and (2) evaluate predictions against ground truth. The
attacker code itself only ever receives observations + transcript + seed labels.

Shared defaults (device count, tier count K, dataset, seed) live in
experiments/config.py so every experiment is mutually consistent unless it is
directly studying one of those parameters.
"""