#!/usr/bin/env python
"""Minimal offline test runner (pytest-compatible subset).

The project's tests are written for pytest. This runner executes them WITHOUT
pytest installed -- it supports the subset of pytest features the suite uses:
plain test_* functions, module-level @pytest.fixture (function-scoped and a
trivial session scope), and @pytest.mark.parametrize. On a machine with pytest
installed, just run ``pytest`` instead; this exists for offline/CI parity and as
an artifact convenience.

Usage:  python run_tests.py [tests_dir]
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))


# ---- a tiny pytest shim so test modules can `import pytest` ----
class _Mark:
    def __getattr__(self, _name):
        def deco(*dargs, **dkw):
            # parametrize records argnames/argvalues on the function
            def wrap(fn):
                if _name == "parametrize":
                    argnames = [a.strip() for a in dargs[0].split(",")]
                    cases = getattr(fn, "_parametrize", [])
                    cases.append((argnames, list(dargs[1])))
                    fn._parametrize = cases
                return fn
            # markers used as bare decorators (e.g. @pytest.mark.slow) get no args
            if len(dargs) == 1 and callable(dargs[0]) and not dkw:
                return dargs[0]
            return wrap
        return deco


class _ApproxScalar:
    def __init__(self, v, rel=1e-6, abs=1e-12):
        self.v, self.rel, self.abs = v, rel, abs

    def __eq__(self, other):
        return abs(other - self.v) <= max(self.rel * abs(self.v), self.abs)


class _PytestShim:
    mark = _Mark()

    @staticmethod
    def fixture(*fargs, **fkw):
        def deco(fn):
            fn._is_fixture = True
            return fn
        if len(fargs) == 1 and callable(fargs[0]):
            return deco(fargs[0])
        return deco

    @staticmethod
    def approx(v, rel=1e-6, abs=1e-12):
        return _ApproxScalar(v, rel, abs)

    class raises:  # noqa: N801
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            return et is not None and issubclass(et, self.exc)


sys.modules["pytest"] = _PytestShim()


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _conftest_fixtures() -> dict:
    """Provide the two conftest fixtures the separation tests use, offline.

    conftest.py imports pytest at module top (for its own @pytest.fixture
    decorators), which our shim satisfies; we then call the pure scanner it
    defines and expose the fixture VALUES directly.
    """
    cpath = ROOT / "conftest.py"
    if not cpath.exists():
        return {}
    cmod = _load_module(cpath)
    scanner = getattr(cmod, "transitive_dtfl_imports", None)
    if scanner is None:
        return {}
    forbidden = {"dtfl.latent", "dtfl.controller.oracle", "dtfl.learning"}
    # Expose as zero-arg fixture callables returning the value.
    return {
        "import_scanner": (lambda: scanner),
        "forbidden_for_attack": (lambda: forbidden),
        "has_dtfl": (lambda: True),
    }


def _resolve_fixtures(mod):
    fx = {
        name: fn
        for name, fn in inspect.getmembers(mod, inspect.isfunction)
        if getattr(fn, "_is_fixture", False)
    }
    # Merge conftest-provided fixtures (don't override module-local ones).
    for name, fn in _conftest_fixtures().items():
        fx.setdefault(name, fn)
    return fx


def _call_with_fixtures(fn, fixtures, cache):
    kwargs = {}
    for pname in inspect.signature(fn).parameters:
        if pname in fixtures:
            if pname not in cache:
                cache[pname] = _call_with_fixtures(fixtures[pname], fixtures, cache)
            kwargs[pname] = cache[pname]
    return fn(**kwargs)


def run_file(path: Path) -> tuple[int, int, list[str]]:
    mod = _load_module(path)
    fixtures = _resolve_fixtures(mod)
    passed = failed = 0
    failures = []
    tests = [
        (name, fn)
        for name, fn in inspect.getmembers(mod, inspect.isfunction)
        if name.startswith("test_") and not getattr(fn, "_is_fixture", False)
    ]
    for name, fn in tests:
        param_sets = getattr(fn, "_parametrize", None)
        invocations = []
        if param_sets:
            # build the cartesian product of stacked parametrize decorators
            import itertools
            grids = [[(an, v) for v in vals] for an, vals in param_sets]
            for combo in itertools.product(*grids):
                extra = {}
                for argnames, val in combo:
                    if len(argnames) == 1:
                        extra[argnames[0]] = val
                    else:
                        for a, v in zip(argnames, val, strict=True):
                            extra[a] = v
                invocations.append(extra)
        else:
            invocations.append({})

        for extra in invocations:
            cache = {}
            label = name + (f"[{extra}]" if extra else "")
            try:
                sig = inspect.signature(fn)
                kwargs = dict(extra)
                for pname in sig.parameters:
                    if pname in fixtures and pname not in kwargs:
                        if pname not in cache:
                            cache[pname] = _call_with_fixtures(fixtures[pname], fixtures, cache)
                        kwargs[pname] = cache[pname]
                fn(**kwargs)
                passed += 1
            except Exception:
                failed += 1
                failures.append(f"{path.name}::{label}\n{traceback.format_exc()}")
    return passed, failed, failures


def main():
    tests_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests"
    total_p = total_f = 0
    all_failures = []
    for path in sorted(tests_dir.glob("test_*.py")):
        p, f, fails = run_file(path)
        status = "OK" if f == 0 else f"{f} FAILED"
        print(f"  {path.name:<40} {p:>3} passed  {status}")
        total_p += p
        total_f += f
        all_failures += fails
    print("-" * 60)
    print(f"TOTAL: {total_p} passed, {total_f} failed")
    if all_failures:
        print("\n=== FAILURES ===")
        for fail in all_failures:
            print(fail)
    sys.exit(1 if total_f else 0)


if __name__ == "__main__":
    main()
