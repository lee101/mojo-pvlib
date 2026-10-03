"""Shared setup: put `python/` on the path and prove the .so actually loads.

Every test here exercises the compiled library, so the load is made observable:
this autouse fixture checks that `dist/libmojo-pvlib.so` exists and dlopens,
and fails the whole run if it does not. There is no pure-Python fallback -- the
wrappers are thin ctypes shims over `@export`ed kernels -- so a missing build
has to be a hard error rather than a quiet skip. Build it with
`pixi run build`.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _require_compiled_library():
    """Fail loudly, once, if the compiled shared library is absent or unloadable."""
    from mojopvlib import _lib

    path = _lib.LIB
    if not os.path.exists(path):
        pytest.fail(
            f"compiled library missing at {path!r}; run `pixi run build` first",
            pytrace=False,
        )
    try:
        _lib.lib()  # raises OSError if the .so cannot be dlopen'd
    except OSError as exc:  # pragma: no cover - build/toolchain failure
        pytest.fail(f"could not load {path!r}: {exc}", pytrace=False)
    return path