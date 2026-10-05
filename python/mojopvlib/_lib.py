"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOPVLIB_LIB") or os.path.join(ROOT, "dist", "libmojo-pvlib.so")

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    # tools
    "mpv_cosd": ([I, I, I], None),
    "mpv_sind": ([I, I, I], None),
    "mpv_tand": ([I, I, I], None),
    "mpv_asind": ([I, I, I], None),
    "mpv_acosd": ([I, I, I], None),
    "mpv_atand": ([I, I, I], None),
    # spa
    "mpv_sp_tables": ([I], None),
    "mpv_solar_position_loop": (
        [I, I, I, F, F, F, F, F, F, I, I, I, I, I, I, I, I, I], None,
    ),
    "mpv_calculate_deltat": ([I, I, I, I], None),
    # solarposition
    "mpv_calculate_simple_day_angle": ([I, F, I, I], None),
    "mpv_equation_of_time_spencer71": ([I, I, I, I], None),
    "mpv_equation_of_time_pvcdrom": ([I, I, I, I], None),
    "mpv_declination_spencer71": ([I, I, I, I], None),
    "mpv_declination_cooper69": ([I, I, I, I], None),
    "mpv_solar_azimuth_analytical": ([I, I, I, I, I, I], None),
    "mpv_solar_zenith_analytical": ([I, I, I, I, I], None),
    "mpv_hour_angle": ([I, F, I, I, I], None),
    "mpv_sunset_angle": ([F, I, I, I], None),
    # atmosphere
    "mpv_pres2alt": ([I, I, I], None),
    "mpv_alt2pres": ([I, I, I], None),
    "mpv_get_absolute_airmass": ([I, I, I, I], None),
    "mpv_get_relative_airmass": ([I, I, I, I], None),
    "mpv_gueymard94_pw": ([I, I, I, I], None),
    "mpv_bird_hulstrom80_aod_bb": ([I, I, I, I], None),
    "mpv_kasten96_lt": ([I, I, I, I, I], None),
    "mpv_rh_from_tdew": ([I, I, I, I, F, F, F], None),
    "mpv_tdew_from_rh": ([I, I, I, I, F, F, F], None),
    "mpv_angstrom_aod_at_lambda": ([I, I, I, I, I, I], None),
    "mpv_angstrom_alpha": ([I, I, I, I, I, I], None),
    "mpv_windspeed_powerlaw": ([I, I, I, F, I, I], None),
    # irradiance
    "mpv_get_extra_radiation": ([I, I, I, F, I], None),
    "mpv_aoi_projection": ([I, I, I, I, I, I], None),
    "mpv_aoi": ([I, I, I, I, I, I], None),
    "mpv_beam_component": ([I, I, I, I, I, I, I], None),
    "mpv_get_ground_diffuse": ([I, I, I, I, F], None),
    "mpv_isotropic": ([I, I, I, I, I], None),
    "mpv_klucher": ([I, I, I, I, I, I, I, I], None),
    "mpv_haydavies": ([I] * 13, None),
    "mpv_reindl": ([I] * 13, None),
    "mpv_king": ([I, I, I, I, I, I], None),
    "mpv_perez": ([I] * 15, None),
    "mpv_clearsky_index": ([I, I, I, I, F], None),
    "mpv_clearness_index": ([I, I, I, I, I, F, F], None),
    "mpv_clearness_index_zenith_independent": ([I, I, I, I, F], None),
    "mpv_disc": ([I, I, I, I, I, I, I, F, F, F, F, I], None),
    "mpv_dirint": ([I, I, I, I, I, I, I, I, F, F, I, I, I, F, F, I], None),
    "mpv_dirindex": ([I, I, I, I, I], None),
    "mpv_erbs": ([I, I, I, I, I, I, I, F, F], None),
    "mpv_orgill_hollands": ([I, I, I, I, I, I, I, F, F], None),
    "mpv_boland": ([I, I, I, I, I, I, I, F, F, F, F], None),
    "mpv_dni": ([I, I, I, I, I, I, F, I, F, F], None),
    "mpv_louche": ([I, I, I, I, I, I, I, F], None),
    "mpv_campbell_norman": ([I, I, I, I, I, F, F, F], None),
    "mpv_diffuse_par_spitters": ([I, I, I, I], None),
    "mpv_poa_components": ([I, I, I, I, I, I, I, I], None),
    # tables
    "mpv_dirint_coeffs_table": ([I], None),
    "mpv_perez_coefficients": ([I, I, I], None),
}

SPA_TABLE_NAMES = (
    "L0", "L1", "L2", "L3", "L4", "L5", "B0", "B1",
    "R0", "R1", "R2", "R3", "R4", "NUTATION_ABCD_ARRAY", "NUTATION_YTERM_ARRAY",
)
SPA_TABLE_COLUMNS = (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5)

AIRMASS_MODELS = {
    "kastenyoung1989": 0,
    "kasten1966": 1,
    "simple": 2,
    "pickering2002": 3,
    "youngirvine1967": 4,
    "young1994": 5,
    "gueymard1993": 6,
    "gueymard2003": 7,
}

EXTRA_RADIATION_METHODS = {"asce": 0, "spencer": 1}

PEREZ_MODELS = (
    "allsitescomposite1990", "allsitescomposite1988", "sandiacomposite1988",
    "usacomposite1988", "france1988", "phoenix1988", "elmonte1988",
    "osage1988", "albuquerque1988", "albany1988",
)


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOPVLIB_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOPVLIB_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-pvlib.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a) -> np.ndarray:
    """A C-contiguous float64 buffer holding the elements of `a`.

    `np.ravel` answers the C-order question for the original dtype, so an
    object array holding strings or dates raises here rather than later.
    """
    a = np.asarray(a)
    if a.dtype == object:
        raise TypeError(
            f"mojopvlib works on float64 arrays, got dtype {a.dtype!r}"
        )
    return np.ascontiguousarray(np.ravel(a), dtype=np.float64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def out(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def is_series(a) -> bool:
    """True for a pandas Series, the only input whose type pvlib preserves."""
    import pandas as pd
    return isinstance(a, pd.Series)


def broadcast(a, n: int) -> np.ndarray:
    """`a` as a float64 buffer of length `n`.

    A single value is repeated across the series. Any other length is a
    mismatch numpy would reject, and the kernel would read past the end of
    the buffer, so it raises here instead.
    """
    a = f64(a)
    if a.size == n:
        return a
    if a.size == 1:
        return np.full(n, a[0])
    raise ValueError(
        f"operands could not be broadcast together with shapes "
        f"({a.size},) ({n},) "
    )


def _kernel_scalar(value, name="argument"):
    """A kernel's scalar argument, as numpy's scalar arithmetic would read it.

    A number passes through; an array argument is a length-1 operand and its
    value is taken, exactly as `number * array` reads `float(number)`. A longer
    array cannot be honoured by a scalar slot in the ABI, so it is refused
    rather than quietly truncated.
    """
    if np.ndim(value) == 0:
        return float(value)
    flat = np.ravel(value)
    if flat.size != 1:
        raise TypeError(
            f"only length-1 arrays can be used as a scalar {name}, "
            f"got {flat.size}"
        )
    return float(flat[0])


def vectors(*values):
    """Broadcast array-likes to a common length, the way numpy would.

    Returns `(n, arrays)`; a scalar or 0-d argument is materialised as a
    full-length buffer, a `None` stays `None` for the optional kernel
    arguments. Lengths that numpy would refuse to broadcast raise here
    rather than being read out of bounds by the kernel.

    The common length is the product of the operands' shapes, not of their
    per-axis dimensions: the ABI is one flat float64 buffer per argument and
    every kernel here is elementwise, so an n-dimensional input is flattened
    and `shaped` puts the broadcast shape back on the way out.
    """
    sizes = [np.size(v) for v in values if v is not None and not np.isscalar(v)]
    distinct = {s for s in sizes if s != 1}
    if len(distinct) > 1:
        shapes = " ".join(f"{np.shape(v)}" for v in values if v is not None)
        raise ValueError(
            f"operands could not be broadcast together with shapes {shapes} "
        )
    n = max(sizes) if sizes else 1
    arrays = []
    for v in values:
        if v is None:
            arrays.append(None)
        else:
            arrays.append(broadcast(v, n))
    return n, arrays


def shaped(res: np.ndarray, *inputs):
    """A flat kernel result back in the shape and container numpy would return.

    The ABI is one flat float64 buffer per argument, so an n-dimensional input
    is flattened on the way in and has to be reshaped on the way out. The
    shape is numpy's own broadcast of the arguments, and a Series argument
    that already matches it brings its index and name along.

    The scalar case stays a numpy scalar rather than a Python float, because
    that is what a numpy ufunc hands back for a scalar argument and
    `np.float64(3.0) != 3.0` under `isinstance` is a difference a caller can
    see.
    """
    args = [v for v in inputs if v is not None]
    if all(np.ndim(v) == 0 for v in args):
        return res[0]
    shape = np.broadcast_shapes(*[np.shape(v) for v in args])
    import pandas as pd
    for like in inputs:
        # Upstream reaches most of these answers through a numpy ufunc, and a
        # ufunc hands back the input's container: a Series stays a Series and a
        # pandas Index stays an Index. Only the callers that want the unwrapped
        # array strip it afterwards.
        if not isinstance(like, (pd.Series, pd.Index)):
            continue
        if np.ndim(like) != len(shape):
            continue
        if isinstance(like, pd.Series):
            return pd.Series(res.reshape(shape), index=like.index, name=like.name)
        return pd.Index(res.reshape(shape), name=like.name)
    return res.reshape(shape)


_spa_tables = None


def spa_table(name: str) -> np.ndarray:
    """One NREL SPA coefficient table, shaped like upstream's `spa.L0` etc.

    Mojo has no module-level mutable state, so `spa_tables` writes the whole set
    into one host-owned buffer; this hands back a view of the requested table.
    """
    global _spa_tables
    if _spa_tables is None:
        buf = np.zeros(4096, dtype=np.float64)
        lib().mpv_sp_tables(addr(buf))
        n_tables = int(buf[0])
        if n_tables != len(SPA_TABLE_NAMES):
            raise RuntimeError("spa table count mismatch")
        rows = buf[1:1 + n_tables].astype(np.int64)
        start = 1 + n_tables
        spans = rows * np.array(SPA_TABLE_COLUMNS[:n_tables], dtype=np.int64)
        offsets = start + np.concatenate(([0], np.cumsum(spans)[:-1]))
        _spa_tables = (buf, rows, offsets)
    buf, rows, offsets = _spa_tables
    k = SPA_TABLE_NAMES.index(name)
    cols = SPA_TABLE_COLUMNS[k]
    start, stop = int(offsets[k]), int(offsets[k] + rows[k] * cols)
    return buf[start:stop].reshape(int(rows[k]), cols)


def spa_tables_buffer() -> np.ndarray:
    """The buffer the SPA coefficient tables were written into."""
    spa_table("L0")
    return _spa_tables[0]


_dirint_coeffs = None


def dirint_coeffs() -> np.ndarray:
    """The DIRINT coefficient cube, [kt_prime, zenith, delta_kt_prime, w]."""
    global _dirint_coeffs
    if _dirint_coeffs is None:
        buf = np.zeros(6 * 6 * 7 * 5, dtype=np.float64)
        lib().mpv_dirint_coeffs_table(addr(buf))
        _dirint_coeffs = buf.reshape(6, 6, 7, 5)
    return _dirint_coeffs


def _perez_model_id(model: str) -> int:
    name = model.lower()
    if name == "1990":
        name = "allsitescomposite1990"
    if name not in PEREZ_MODELS:
        raise ValueError(f"invalid perez model {model}")
    return PEREZ_MODELS.index(name)


def perez_coefficient_table(model: str):
    """The 9-row table the kernel reads: the all-NaN row upstream appends for
    invalid clearness bins sits at index 0, the real coefficients follow."""
    f1 = np.zeros((9, 3), dtype=np.float64)
    f2 = np.zeros((9, 3), dtype=np.float64)
    lib().mpv_perez_coefficients(_perez_model_id(model), addr(f1), addr(f2))
    return f1, f2


def perez_coefficients(model: str):
    """`(F1coeffs, F2coeffs)` for one of pvlib's Perez coefficient sets."""
    f1, f2 = perez_coefficient_table(model)
    return f1[1:].copy(), f2[1:].copy()


def main() -> int:
    """`python -m mojopvlib._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
