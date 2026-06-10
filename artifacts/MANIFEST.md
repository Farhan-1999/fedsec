# Artifact Manifest

This manifest records the seeds, configurations, and outputs for every experiment,
so each result and figure is traceable and reproducible. Regenerate everything with
`python reproduce.py all` from the repository root.

## Determinism

Every stochastic draw is threaded through a seeded `RngHub` (see `src/dtfl/rng.py`),
with sub-streams seeded by hashing their name. Consequences:

- A given base seed reproduces the entire latent population AND the entire
  transcript byte-for-byte (tested in `tests/test_transcript_sim.py::test_determinism_identical_transcript`).
- Adding a new component never perturbs existing components' draws (name-hashed
  sub-streams), so results remain reproducible as the code evolves.
- Multi-seed replicates use deterministic `(engine_seed, attack_seed)` pairs from
  `experiments/multiseed.py::seed_pairs`, so an S=5 run reproduces the first 3 of
  an S=3 run exactly.

## Default model parameters (single source: `src/dtfl/latent/config.py`)

| Parameter | Symbol | Default |
|---|---|---|
| capability classes | C | 5 |
| population mixture | pi | (.10,.25,.30,.25,.10) |
| class mean spacing (log) | Delta_class | 0.40 |
| within-class spread (log) | s_within | 0.15 |
| proxy noise (log) | eta | 0.20 |
| -> resulting SNR | | 2.56 |
| heavy-tail prob / scale | | 0.05 / 0.50 |

## Experiments and outputs

| Stage | Script | Seed(s) | N / rounds | Output |
|---|---|---|---|---|
| gate | `experiments/run_gate.py` | engine 101, attack 0 | 3000 / 80 | console (Gates 2-4) |
| pareto | `experiments/run_headline_pareto.py` | engine 101, attack 0 | 3000 / 80 | `results/headline_pareto.csv`, `figures/headline_pareto.png` |
| ladder | `experiments/run_ladder.py` | engine 101, attack 0 | 3000 / 80 | `results/ladder.csv` |
| linkability | `experiments/run_linkability.py` | engine 101 | 3000 / 80 | `results/linkability.csv`, `figures/linkability_horizon.png` |
| multiseed frontier | `experiments/run_multiseed_frontier.py` | base 2024, S=5 | 1500 / 60 | `results/multiseed_frontier.csv`, `figures/frontier_errorbars.png` |
| multiseed ladder+link | `experiments/run_multiseed_ladder_linkability.py` | base 2024, S=5 | 1500 / 60 | `results/multiseed_ladder.csv`, `results/multiseed_linkability.csv` |

## Key results (single-seed unless noted)

- **Gate 2 (leakage exists):** undefended L1 capability-inference advantage ~0.64.
- **Headline frontier:** advantage falls 0.72 -> 0.00 as m_min suppression rises;
  transition is a sharp "cliff" because equal-size tiers fail together.
- **Bucket axis is inert:** timing-bucket granularity barely affects capability
  advantage (tier identity carries the signal).
- **Adversary ladder (5 seeds):** L0 ~0.10 (noisy), L1 ~0.60, L2 ~0.60 (L1~L2:
  realistic attacker matches Bayes-optimal), L3 ~0.92 (omniscient ceiling).
- **Linkability (5 seeds):** anonymity collapses with horizon; ~41% uniquely
  identifiable by h=10, ~99% by h=40; accumulation rate ~0.013 (near seed-invariant).
- **Multi-seed frontier:** CIs tight at extremes, wide through the cliff
  (advantage half-width up to +/-0.08) -- the trade-off is high-variance near
  practical operating points.
- **Controllers:** transcript-only adaptive controllers match the latent oracle in
  the stationary case; under drift, EWMA tracks targets ~4x better than a frozen
  fixed policy.

## Known caveats (for honest reporting)

- L0 (unsupervised) is high-variance at S=5; report with its wide CI or use more seeds.
- Convex models converge in ~1 round on IID data (no multi-round curve); learning-
  curve figures use the nonconvex MLP / torch CNN.
- The adversary-observation model (attacker sees a device's tier only on released
  tier-rounds) is load-bearing and must be justified explicitly in the threat model.
