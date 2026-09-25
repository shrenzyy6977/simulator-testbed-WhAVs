"""Minimal test runner for machines without pytest: python tests/run_all.py"""
import importlib
import inspect
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class _Mark:  # tiny stand-in for pytest.mark.parametrize
    @staticmethod
    def parametrize(name, values, ids=None):
        def deco(f):
            f._params = (name, list(values))
            return f
        return deco


try:
    import pytest  # noqa: F401
except ImportError:
    import types
    sys.modules["pytest"] = types.SimpleNamespace(mark=_Mark())

failed = passed = 0
for f in sorted(Path(__file__).parent.glob("test_*.py")):
    mod = importlib.import_module(f.stem)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        cases = [({},)] if not hasattr(fn, "_params") else [({fn._params[0]: v},) for v in fn._params[1]]
        for (kw,) in cases:
            label = f"{f.stem}::{name}{kw if kw else ''}"
            try:
                fn(**kw)
                passed += 1
                print("PASS", label)
            except Exception:
                failed += 1
                print("FAIL", label)
                traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
