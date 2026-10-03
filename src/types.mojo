"""Pointer aliases and numpy-compatible scalar helpers.

The whole library is float64 and caller-allocated: no kernel in this repo
allocates, so nothing here can leak and the host owns every buffer.
"""

from std.math import isnan, exp, log
from max.algorithm import parallelize
from std.sys import num_logical_cores

comptime F64 = Pointer[Float64, AnyOrigin[mut=True]]

# `parallelize` costs a fixed 6-13 ms of thread launch per call whatever the
# input size, so a kernel only clears it once its serial pass is worth more than
# that. These are the element counts at which that happens, measured per
# process against a serial-only build of the same source, one per tier of
# per-element cost.
comptime PAR_MIN_TINY = 1024
comptime PAR_MIN_HEAVY = 1 << 16
comptime PAR_MIN_MEDIUM = 1 << 19
comptime PAR_MIN_CHEAP = 786432


def par_workers() -> Int:
    # One worker per physical core: these kernels are libm-latency bound, so
    # hyperthread siblings buy nothing and oversubscribing costs about 30%.
    return max(num_logical_cores() // 2, 1)


def p(addr: Int) -> F64:
    return F64(unsafe_from_address=addr)


# numpy's maximum/minimum/clip, which propagate NaN; the builtins do not.
def maximum(a: Float64, b: Float64) -> Float64:
    if isnan(a):
        return a
    if isnan(b):
        return b
    return a if a > b else b


def minimum(a: Float64, b: Float64) -> Float64:
    if isnan(a):
        return a
    if isnan(b):
        return b
    return a if a < b else b


def clip(x: Float64, lo: Float64, hi: Float64) -> Float64:
    return minimum(maximum(x, lo), hi)


def nan_() -> Float64:
    return Float64(0.0) / Float64(0.0)


def pow_(base: Float64, e: Float64) -> Float64:
    """`base ** e`, matching numpy where LLVM's `pow` does not.

    Two divergences from libm's `pow` are patched, both verified by probe:

    * Underflow. For a base in (0, 1) with a large positive exponent the true
      result falls below the smallest double and `pow` returns 1.0 rather than
      0.0 or a denormal; numpy returns the true value. Solar models hit this at
      the horizon, where relative airmass reaches ~1.6e16 at zenith 90 deg and
      `campbell_norman`'s `transmittance**airmass` must underflow or DNI comes
      back at the full extraterrestrial value.
    * A NaN exponent. `pow` returns +/-inf for it; numpy returns NaN. The
      airmass is NaN above the horizon, so `campbell_norman` reaches this on
      every below-horizon zenith.

    Everything else is `pow` itself, which agrees with libm to about 5e-9
    relative on the ordinary range -- the same disagreement every other `**` in
    this port carries -- and so is left alone rather than paying for a second
    `log` and `exp` per element.
    """
    if isnan(e):
        return nan_()
    if base < 1.0 and base > 0.0 and e > 0.0:
        var p = base**e
        if p == 1.0:
            return exp(e * log(base))
        return p
    return base**e
