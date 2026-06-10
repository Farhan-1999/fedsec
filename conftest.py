"""Pytest root configuration and shared fixtures.

The most important thing here is the machinery that enforces the threat-model
boundary: code in the ``attack`` package must never be able to reach ground-truth
state. We check this *statically* by parsing the import graph, so the test fails
even if a leaky import is never exercised at runtime.

Forbidden-for-attack modules:
- ``dtfl.latent``           : ground truth (capability classes, true latencies)
- ``dtfl.controller.oracle``: the oracle controller legitimately reads latent
                              state; the attacker must not reach it transitively.
- ``dtfl.learning``         : holds per-client data/updates pre-aggregation.

The scanner walks the transitive first-party import closure of a starting
package and returns every ``dtfl.*`` module it can reach. A separation test then
asserts the forbidden set is disjoint from that closure.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parent / "src"


def _module_file(modname: str) -> Path | None:
    """Resolve a dotted ``dtfl.*`` module name to its source file, if first-party."""
    rel = modname.replace(".", "/")
    candidates = [SRC_ROOT / f"{rel}.py", SRC_ROOT / rel / "__init__.py"]
    for c in candidates:
        if c.exists():
            return c
    return None


def _imports_in_file(path: Path) -> set[str]:
    """Return the set of dotted module names imported by a single source file.

    Handles ``import a.b`` and ``from a.b import c``. For ``from`` imports we
    record the module being imported from; we also record ``module.name`` so a
    ``from dtfl.controller import oracle`` is caught as reaching
    ``dtfl.controller.oracle``.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import; out of scope for this first-party scan.
                continue
            mod = node.module or ""
            found.add(mod)
            for alias in node.names:
                found.add(f"{mod}.{alias.name}")
    return found


def transitive_dtfl_imports(start_package: str) -> set[str]:
    """Walk the transitive first-party (``dtfl.*``) import closure of a package.

    Returns every ``dtfl.*`` module reachable from ``start_package``. Non-first-
    party imports (numpy, sklearn, ...) are ignored. Submodules of the starting
    package are included as seeds so the whole package is scanned, not just its
    ``__init__``.
    """
    seen: set[str] = set()
    queue: list[str] = []

    # Seed with the package __init__ and every .py submodule under it.
    pkg_dir = SRC_ROOT / start_package.replace(".", "/")
    if pkg_dir.is_dir():
        for py in pkg_dir.rglob("*.py"):
            rel = py.relative_to(SRC_ROOT).with_suffix("")
            modname = ".".join(rel.parts)
            if modname.endswith(".__init__"):
                modname = modname[: -len(".__init__")]
            queue.append(modname)
    else:
        queue.append(start_package)

    while queue:
        mod = queue.pop()
        if mod in seen:
            continue
        seen.add(mod)
        f = _module_file(mod)
        if f is None:
            continue
        for imp in _imports_in_file(f):
            if not imp.startswith("dtfl"):
                continue
            # Normalize: an imported symbol like dtfl.types.TierRecord should
            # also register its parent module dtfl.types.
            parts = imp.split(".")
            for i in range(2, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if _module_file(prefix) is not None and prefix not in seen:
                    queue.append(prefix)
            if imp not in seen:
                queue.append(imp)
    return {m for m in seen if m.startswith("dtfl")}


@pytest.fixture(scope="session")
def import_scanner():
    """Expose the transitive import scanner to tests."""
    return transitive_dtfl_imports


@pytest.fixture(scope="session")
def forbidden_for_attack() -> set[str]:
    """Modules the attack package must never reach (transitively)."""
    return {
        "dtfl.latent",
        "dtfl.controller.oracle",
        "dtfl.learning",
    }


def _ensure_src_on_path() -> None:
    src = str(SRC_ROOT.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


# Make ``dtfl`` importable even when the package isn't pip-installed (editable
# install is recommended, but tests should work from a bare checkout too).
_ensure_src_on_path()


@pytest.fixture
def has_dtfl() -> bool:
    """Skip-guard: is the dtfl package importable at all?"""
    return importlib.util.find_spec("dtfl") is not None
