"""Console hygiene: silence known-harmless third-party warnings.

Imported at the very top of every experiment entry point (before scikit-learn is
used) so the run output stays readable. We suppress ONLY specific, cosmetic
warnings and leave everything else visible, so nothing meaningful is hidden.

Currently filtered:
  - sklearn.utils.parallel delayed/Parallel mismatch: a config-propagation nudge
    emitted per joblib worker dispatch (e.g. by RandomForestClassifier in the L1
    attacker). Harmless; results are unaffected. It floods the console because the
    privacy sweeps fit many forests across m_min x seed x round.
"""
from __future__ import annotations

import warnings


def install() -> None:
    """Register the filters. Safe to call multiple times."""
    warnings.filterwarnings(
        "ignore",
        message=r".*sklearn\.utils\.parallel\.delayed.*",
        category=UserWarning,
    )
    # Belt-and-suspenders: some sklearn versions route this through a module
    # path match rather than the message text.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"sklearn\.utils\.parallel",
    )


# Install on import so `import quiet` alone is enough.
install()