"""pvlib.tools: degree-argument trigonometry.

Same names, same argument order and same return conventions as
`pvlib.tools`; the work happens in `src/tools.mojo`.
"""

from __future__ import annotations

from ._lib import addr, lib, out, shaped, vectors

_ELEMWISE = {
    "cosd": "mpv_cosd",
    "sind": "mpv_sind",
    "tand": "mpv_tand",
    "asind": "mpv_asind",
    "acosd": "mpv_acosd",
    "atand": "mpv_atand",
}


def _elementwise(symbol: str, angle):
    """Run one of the degree-argument trig kernels over `angle`."""
    n, (x,) = vectors(angle)
    res = out(n)
    getattr(lib(), symbol)(addr(x), addr(res), n)
    return shaped(res, angle)


def cosd(angle):
    """
    Trigonometric cosine with angle input in degrees.
    """
    return _elementwise(_ELEMWISE["cosd"], angle)


def sind(angle):
    """
    Trigonometric sine with angle input in degrees.
    """
    return _elementwise(_ELEMWISE["sind"], angle)


def tand(angle):
    """
    Trigonometric tangent with angle input in degrees.
    """
    return _elementwise(_ELEMWISE["tand"], angle)


def asind(number):
    """
    Trigonometric inverse sine returning an angle in degrees.
    """
    return _elementwise(_ELEMWISE["asind"], number)


def acosd(number):
    """
    Trigonometric inverse cosine returning an angle in degrees.
    """
    return _elementwise(_ELEMWISE["acosd"], number)


def atand(number):
    """
    Trigonometric inverse tangent returning an angle in degrees.
    """
    return _elementwise(_ELEMWISE["atand"], number)
