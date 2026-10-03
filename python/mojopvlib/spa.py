"""pvlib.spa: the NREL Solar Position Algorithm, ported to Mojo.

`pvlib/solarposition.py` calls into this module; the kernels live in
`src/spa.mojo` and the coefficient tables in `src/spa_tables.mojo`.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, broadcast, lib, out, shaped, spa_tables_buffer, vectors


def julian_day(unixtime):
    """Julian day from a unix/epoch timestamp, in seconds."""
    return np.asarray(unixtime, dtype=np.float64) * 1.0 / 86400 + 2440587.5


def solar_position_loop(unixtime, delta_t, loc_args, out_):
    """Loop through the time array and calculate the solar position"""
    lat, lon, elev, pressure, temp, atmos_refract, sst, esd = (
        float(v) for v in loc_args
    )
    n, (utime, dt) = vectors(unixtime, delta_t)
    res = np.zeros((6, n), dtype=np.float64)
    lib().mpv_solar_position_loop(
        addr(utime), addr(dt), n,
        lat, lon, elev, pressure, temp, atmos_refract,
        int(sst), int(esd),
        addr(res[0]), addr(res[1]), addr(res[2]), addr(res[3]), addr(res[4]),
        addr(res[5]), addr(spa_tables_buffer()),
    )
    if out_ is None:
        return res
    out_[:, :] = res
    return out_


def solar_position(unixtime, lat, lon, elev, pressure, temp, delta_t,
                   atmos_refract, numthreads=8, sst=False, esd=False):
    """Calculate the solar position for an array of unix timestamps.

    Returns the `(6, n)` result array whose rows are the apparent zenith,
    zenith, apparent elevation, elevation, azimuth and equation of time.
    `numthreads` is accepted for API compatibility and ignored: there is no
    numba path, the kernel is called once for the whole array.
    """
    loc_args = [lat, lon, elev, pressure, temp, atmos_refract, sst, esd]
    return solar_position_loop(unixtime, delta_t, loc_args, None)


def earthsun_distance(unixtime, delta_t, numthreads):
    """
    Calculates the distance from the earth to the sun using the
    NREL SPA algorithm described in [1].

    References
    ----------
    [1] Reda, I., Andreas, A., 2003. Solar position algorithm for solar
    radiation applications. Technical report: NREL/TP-560-34302.
    """
    R = solar_position(unixtime, 0, 0, 0, 0, 0, delta_t, 0, numthreads, esd=True)[0]
    return R


def calculate_deltat(year, month):
    """Calculate the difference between Terrestrial Dynamical Time (TD)
    and Universal Time (UT).

    Equations taken from http://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html
    """
    year = np.asarray(year, dtype=np.float64)
    month = np.asarray(month, dtype=np.float64)
    scalar = year.ndim == 0 and month.ndim == 0
    n = max(year.size, month.size)
    y = broadcast(year, n)
    m = broadcast(month, n)
    deltat = out(n)
    lib().mpv_calculate_deltat(addr(y), addr(m), addr(deltat), n)
    if scalar:
        return float(deltat[0])
    return shaped(deltat, year, month)
