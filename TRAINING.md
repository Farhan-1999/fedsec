# Training and Learning Curves

This document covers running **real federated training** under the timing/tier
dynamics and producing the four learning curves: loss, accuracy, convergence, and
training-time. The privacy experiments (gate, frontier, ladder, linkability) do
not need any of this — training is the **utility** side of the study.

## Quick start (no downloads)

```bash
python reproduce.py training
```

This runs federated training on in-memory synthetic data with the NumPy MLP and
writes:

- `artifacts/results/training_synthetic.csv` — per-round trajectory,
- `artifacts/figures/training_synthetic_curves.png` — the 2×2 curve panel.

Equivalent explicit form:

```bash
python experiments/run_training.py --rounds 80 --hidden 64
python experiments/plots/training_curves.py training_synthetic
```

## The four curves

The plot panel contains:

| Panel | Curve | x-axis | What it shows |
|---|---|---|---|
| top-left | accuracy / convergence | round | validation accuracy climbing then plateauing |
| top-right | loss | round | validation cross-entropy falling then flattening |
| bottom-left | training-time (measured) | wall-clock seconds | accuracy vs real compute time spent |
| bottom-right | time-to-accuracy (systems) | simulated time (deadline units) | accuracy vs protocol wall-clock — the standard FL systems metric |

"Convergence" is the accuracy/loss-vs-round shape (watch it flatten). The two
time curves answer different questions: measured wall-clock = "how long did my
experiment take"; time-to-accuracy = "how efficient is the protocol per unit of
simulated deadline time."

The CSV columns are `run, round, val_accuracy, val_loss, virtual_time,
wall_time, participation`, so you can plot any custom view from it directly.

## Flags (`run_training.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `synthetic` | `synthetic`, `cifar10`, or `cifar100` |
| `--rounds` | `80` | number of FL rounds |
| `--hidden` | `64` | hidden-layer width (0 = linear softmax) |
| `--lr` | `0.05` | client local learning rate |
| `--devices` | `300` | client population size |
| `--seed` | `42` | engine seed |
| `--compare` | off | overlay multiple controllers on every curve |

## Real data (CIFAR-10 / CIFAR-100)

Real datasets need the optional `learning` extra (installs torch + torchvision):

```bash
pip install -e ".[learning]"
python experiments/run_training.py --data cifar10 --rounds 200
python experiments/plots/training_curves.py training_cifar10
```

CIFAR downloads automatically to `./data` on first run (needs internet that once).
The driver switches to the torch CNN automatically for non-synthetic data. Expect:

- a longer, more realistic convergence climb (more accuracy headroom than synthetic),
- much larger measured wall-clock (real conv training per client) — which is what
  makes the measured-time curve meaningful.

FEMNIST is not in torchvision; obtain it via LEAF
(https://github.com/TalwalkarLab/leaf), run its `preprocess.sh`, and feed the
pooled arrays through `dtfl.learning.Dataset` + `iid_shard` (the same path the
CIFAR loader uses). Treat this as optional.

## Comparing controllers on the curves

```bash
python experiments/run_training.py --compare
python experiments/plots/training_curves.py training_synthetic_compare
```

Overlays fixed-quantile vs adaptive-quantile deadline controllers on all four
panels, so you can see how the deadline policy affects convergence speed and
time-to-accuracy.

## Notes / expected behavior

- **Synthetic data plateaus around ~0.5 accuracy** by design (it is deliberately
  hard-but-bounded with a small MLP). Use it for fast pipeline checks; use CIFAR
  for publication-quality curves with real headroom.
- **A linear softmax model (`--hidden 0`) converges in ~1 round on IID data** and
  shows an essentially flat curve — that is correct FL behavior, not a bug. Use
  the MLP (`--hidden > 0`) or the torch CNN to get a meaningful multi-round curve.
- Training is **not** part of `reproduce.py all` by default scope expectations for
  the privacy results, but `python reproduce.py training` runs it as its own stage.
- Every run is seeded; re-running reproduces identical trajectories.
```
