# Privacy-Preserving Response-Time-Tiered Federated Learning

Research artifact for a study of **capability leakage and re-identification through
the timing/tier metadata channel** in deadline-tiered federated learning, and the
privacy-utility cost of defending against it.

Secure aggregation hides update *contents*; this work measures what the *metadata*
still leaks. In a system where devices self-assign to tiers by response time (a
proxy for capability) and the server observes only aggregate, tier-level signals,
we show that an honest-but-curious server can (a) infer individual devices'
capability classes and (b) re-identify devices across rounds — and we quantify how
much suppression/coarsening defenses reduce this, at what utility cost.

## Threat model (one paragraph)

The server is honest-but-curious and observes only the **legal transcript**: per
released tier-round, the secure-aggregate, the active-set count, a coarse
release-time bucket, and a success/failure flag. It never sees individual
device→tier mappings, per-device timing, or update contents. The adversary attacks
this metadata channel. An attacker observing a device learns the device's tier only
on **released** tier-rounds (suppressed tiers leak nothing attributable). The hidden
target is each device's latent **capability class**; data is IID, so the only
leakage channel is timing/tier, not data distribution.

## Install

```bash
# core (simulator, attacks, metrics, controllers) — no GPU needed
pip install -e .

# optional: real model training (CIFAR/FEMNIST via torch/torchvision)
pip install -e ".[learning]"

# optional: real secure-aggregation calibration
pip install -e ".[crypto]"

# tests + linting
pip install -e ".[dev]"
```

The core package has **no torch dependency**; the entire privacy pipeline and the
go/no-go gate run on NumPy alone. A pure-NumPy reference model lets training run
without a GPU; the torch model (identical interface) is for the real-data results.

## Reproduce

```bash
python reproduce.py all          # all results + figures
python reproduce.py gate         # just the go/no-go gate
python reproduce.py pareto       # headline frontier + figure
python reproduce.py multiseed    # multi-seed hardening + error-bar figure
```

Outputs land in `artifacts/results/` (CSVs) and `artifacts/figures/` (PNGs). Every
run is seeded; see `artifacts/MANIFEST.md` for seeds, configs, and provenance. The
determinism property (same seed → identical transcript) is enforced by a test.

## Tests

```bash
pytest                  # with pytest installed
python run_tests.py     # offline fallback (no pytest needed)
```

65 tests across all layers. The most important are the **separation tests**
(`tests/test_separation_*.py`): they statically prove the attack package cannot
import ground-truth state (`dtfl.latent`, `dtfl.controller.oracle`, `dtfl.learning`)
and that the transcript record exposes only whitelisted fields. The privacy results
are only meaningful because this boundary holds.

## Repository layout

```
src/dtfl/
  rng.py, types.py        # seeded randomness; the latent/transcript type split
  latent/                 # GROUND TRUTH (capability classes, true latencies, drift)
  protocol/               # tiering, dropout, threshold (t<=m_min), release, merge
  transcript/             # the LEGAL observation set: the attacker's only input
  defense/                # privacy knobs: m_min, count noise, buckets, jitter, padding
  attack/                 # adversaries L0-L3 (import transcript/metrics ONLY)
  metrics/                # capability advantage; anonymity-set / linkability
  controller/             # deadline policies (fixed/quantile/ewma; oracle quarantined)
  learning/               # real FL training (NumPy + torch models, IID resharding)
  sim/                    # orchestration: round, engine, go/no-go gates
experiments/              # run scripts + plots (read latent logs for evaluation)
tests/                    # pytest suite incl. the separation invariants
artifacts/                # results, figures, MANIFEST
```

The directory boundary **is** the threat model: `attack/` physically cannot reach
`latent/`, `controller/oracle.py`, or `learning/`, and this is checked statically.

## Key results

| Result | Value |
|---|---|
| Undefended capability-inference advantage (L1) | ~0.64 |
| Strongest defense | advantage → 0 at high utility cost |
| Adversary ladder | L0 ~0.10 < L1 ~0.60 ≈ L2 ~0.60 < L3 ~0.92 |
| Anonymity collapse | ~41% uniquely identifiable by 10 rounds, ~99% by 40 |
| Timing-bucket defense | inert — tier identity carries the signal |
| Adaptive vs oracle controller | match in stationary case; EWMA ~4× better than fixed under drift |

See `artifacts/MANIFEST.md` for the full result set, the parameter defaults, and the
honest caveats (L0 variance, convex-model convergence, the load-bearing
adversary-observation assumption).

## License

MIT (see `LICENSE`).
