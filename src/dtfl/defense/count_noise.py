"""Count coarsening / noising.

Reveals the active count at reduced fidelity. This is the knob with the explicit
PRIVACY-UTILITY COUPLING: the active count is the merge weight (preamble Section
7), so perturbing the revealed count biases the global update. We therefore
return BOTH the count the transcript shows AND the count the merge should use,
and let the experiment decide whether the merge uses the true or the perturbed
count -- the gap is the utility cost measured by metrics/merge_error.

Modes (from DefenseConfig.CountMode):
  EXACT   : reveal n exactly (no effect).
  HIDDEN  : reveal None. The attacker loses the count channel; the merge must
            fall back to an unweighted/edge-case rule (max bias).
  ROUNDED : reveal n rounded to a multiple. Coarse count-flow signal.
  NOISED  : reveal n + bounded integer noise. DP-flavored; weakens count-flow
            linking and tier-mass estimation.

Hypothesized attack effect: weakens the count-flow linker and class-mass
estimation (adversary spec Section 5). Utility cost: merge-weight bias.
"""

from __future__ import annotations

import numpy as np

from dtfl.defense.config import CountMode, DefenseConfig

__all__ = ["apply_count_defense"]


def apply_count_defense(
    true_count: int,
    config: DefenseConfig,
    rng: np.random.Generator,
) -> int | None:
    """Return the count to REVEAL in the transcript (None if hidden).

    The merge should continue to use ``true_count`` for correctness unless the
    experiment is specifically studying the merge bias from trusting the revealed
    count; that choice lives in the round/merge wiring, not here.
    """
    mode = config.count_mode
    if mode is CountMode.EXACT:
        return true_count
    if mode is CountMode.HIDDEN:
        return None
    if mode is CountMode.ROUNDED:
        m = max(1, config.count_round_to)
        # Round to nearest multiple of m, but never below the m_min floor logic
        # (rounding down through m_min is handled by the release gate separately).
        return int(round(true_count / m) * m)
    if mode is CountMode.NOISED:
        if config.count_noise_scale <= 0:
            return true_count
        noise = int(round(rng.normal(0.0, config.count_noise_scale)))
        return max(0, true_count + noise)
    raise ValueError(f"unknown count mode {mode}")


def merge_weight_for(
    true_count: int,
    revealed_count: int | None,
    trust_revealed: bool,
) -> int:
    """The weight the merge actually uses.

    If ``trust_revealed`` (the realistic case: the server only HAS the revealed
    count), the merge uses the perturbed value, incurring bias. If False (the
    server somehow retains the true count), no bias -- used as the bias-free
    reference in merge-error measurement.
    """
    if not trust_revealed:
        return true_count
    if revealed_count is None:
        # Hidden count: the server has no weight; falls back to weight 1 (treat
        # the tier sum as a single pseudo-update). This is the max-bias fallback.
        return 1
    return max(1, revealed_count)
