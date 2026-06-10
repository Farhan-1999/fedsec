"""Capability-inference advantage (adversary spec Section 7, Goal A).

Advantage normalizes attacker accuracy against the majority-class prior so that
0 = no better than guessing the most common class, 1 = perfect recovery:

    Adv_A = (accuracy - prior) / (1 - prior)

where ``prior`` is the majority-class fraction in the evaluated set. Reported per
adversary rung and per defense configuration. Negative values (worse than the
prior) are clamped to 0 -- an attacker can always fall back to the prior, so
below-prior performance is not a meaningful "negative advantage".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AdvantageResult", "capability_advantage"]


@dataclass
class AdvantageResult:
    accuracy: float
    majority_prior: float
    advantage: float
    per_class_recall: dict[int, float]
    confusion: np.ndarray  # rows = true class, cols = predicted class
    class_labels: list[int]

    def summary(self) -> str:
        pc = "  ".join(f"c{c}:{r:.2f}" for c, r in sorted(self.per_class_recall.items()))
        return (
            f"acc={self.accuracy:.4f}  prior={self.majority_prior:.4f}  "
            f"ADV={self.advantage:.4f}\n  per-class recall: {pc}"
        )


def capability_advantage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> AdvantageResult:
    """Compute normalized capability-inference advantage and a breakdown.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted capability classes for the evaluated (query) devices.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    n = y_true.size
    if n == 0:
        return AdvantageResult(0.0, 0.0, 0.0, {}, np.zeros((0, 0)), [])

    accuracy = float((y_true == y_pred).mean())

    # Majority-class prior over the evaluated set.
    counts = np.bincount(y_true)
    prior = float(counts.max() / n)

    advantage = 0.0 if prior >= 1.0 else max(0.0, (accuracy - prior) / (1.0 - prior))

    # Per-class recall and confusion.
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    label_to_idx = {c: i for i, c in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[label_to_idx[int(t)], label_to_idx[int(p)]] += 1

    per_class_recall: dict[int, float] = {}
    for c in labels:
        i = label_to_idx[c]
        row_total = confusion[i].sum()
        per_class_recall[c] = float(confusion[i, i] / row_total) if row_total > 0 else 0.0

    return AdvantageResult(
        accuracy=accuracy,
        majority_prior=prior,
        advantage=advantage,
        per_class_recall=per_class_recall,
        confusion=confusion,
        class_labels=labels,
    )
