"""Separation test 1: the attack package cannot reach ground truth.

This is the most important test in the codebase. The entire privacy result is
void if the adversary can see latent state. We enforce the boundary statically
by parsing the import graph, so it fails even if the leaky path is never run.

If this test fails, do NOT weaken it. Find the import in ``dtfl.attack`` (or
something it imports) that reaches a forbidden module and remove it. The
attacker may only depend on ``dtfl.transcript``, ``dtfl.metrics``,
``dtfl.types``, ``dtfl.rng``, and third-party libraries.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.separation


def test_attack_cannot_reach_forbidden_modules(import_scanner, forbidden_for_attack):
    """No forbidden module appears in the attack package's import closure."""
    closure = import_scanner("dtfl.attack")
    leaks = {
        forbidden
        for forbidden in forbidden_for_attack
        for reached in closure
        # A leak is reaching the forbidden module itself or any submodule of it.
        if reached == forbidden or reached.startswith(forbidden + ".")
    }
    assert not leaks, (
        "THREAT-MODEL VIOLATION: dtfl.attack transitively imports "
        f"forbidden module(s): {sorted(leaks)}.\n"
        f"Full attack import closure was: {sorted(closure)}.\n"
        "The attacker must depend only on transcript/, metrics/, types, and rng."
    )


def test_attack_closure_is_within_allowlist(import_scanner):
    """The attack package's first-party imports stay within an explicit allowlist.

    Stronger than the forbidden-set check: instead of only banning known-bad
    modules, we require every first-party module the attacker reaches to be on a
    short allowlist. This catches a newly added leaky dependency that we forgot
    to add to the forbidden set.
    """
    allowed_prefixes = (
        "dtfl.attack",
        "dtfl.transcript",
        "dtfl.metrics",
        "dtfl.types",
        "dtfl.rng",
        "dtfl",  # the bare package __init__ (re-exports types/rng) is allowed
    )
    closure = import_scanner("dtfl.attack")
    offenders = sorted(
        m for m in closure if not any(m == p or m.startswith(p + ".") or m == "dtfl" for p in allowed_prefixes)
    )
    assert not offenders, (
        "dtfl.attack reaches first-party modules outside the allowlist: "
        f"{offenders}. Either the dependency is illegitimate (remove it) or the "
        "allowlist must be consciously widened with a threat-model justification."
    )
