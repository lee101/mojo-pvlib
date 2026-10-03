"""pvlib.tools: degree-argument trigonometry.

Port of the `cosd` / `sind` / `tand` / `asind` / `acosd` / `atand` family from
`pvlib/tools.py`, in upstream's order. Upstream dispatches on numpy scalars vs
arrays; here every kernel is a straight-line elementwise map over a buffer, and
the host picks a 1-element buffer for a scalar.
"""

from std.math import sin, cos, tan, asin, acos, atan, pi
from types import F64, parallelize, par_workers, PAR_MIN_CHEAP

# np.radians / np.degrees, which std.math does not provide. numpy scales by
# the constant pi/180 in one rounding, so this does too.
comptime DEG2RAD = pi / 180.0
comptime RAD2DEG = 180.0 / pi


def radians_(x: Float64) -> Float64:
    return x * DEG2RAD


def degrees_(x: Float64) -> Float64:
    return x * RAD2DEG


def cosd_(angle: F64, res: F64, n: Int):
    def one(i: Int) {angle, res}:
        res.unsafe_store(i, cos(radians_(angle.unsafe_load(i))))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def sind_(angle: F64, res: F64, n: Int):
    def one(i: Int) {angle, res}:
        res.unsafe_store(i, sin(radians_(angle.unsafe_load(i))))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def tand_(angle: F64, res: F64, n: Int):
    def one(i: Int) {angle, res}:
        res.unsafe_store(i, tan(radians_(angle.unsafe_load(i))))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def asind_(number: F64, res: F64, n: Int):
    def one(i: Int) {number, res}:
        res.unsafe_store(i, degrees_(asin(number.unsafe_load(i))))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def acosd_(number: F64, res: F64, n: Int):
    def one(i: Int) {number, res}:
        res.unsafe_store(i, degrees_(acos(number.unsafe_load(i))))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def atand_(number: F64, res: F64, n: Int):
    def one(i: Int) {number, res}:
        res.unsafe_store(i, degrees_(atan(number.unsafe_load(i))))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())
