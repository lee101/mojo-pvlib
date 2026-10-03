"""pvlib.solarposition: solar position calculators.

Same names, same argument order and same defaults as
`pvlib.solarposition`; the arithmetic lives in `src/solarposition.mojo` and
`src/spa.mojo`.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import spa
from .atmosphere import alt2pres, pres2alt
from ._lib import addr, is_series, lib, out, shaped, vectors


def _datetime_to_unixtime(dtindex):
    # convert a pandas datetime index to unixtime, making sure to handle
    # different pandas units (ns, us, etc) and time zones correctly
    if dtindex.tz is not None:
        # epoch is 1970-01-01 00:00 UTC, but we need to match the input tz
        # for compatibility with older pandas versions (e.g. v1.3.5)
        epoch = pd.Timestamp("1970-01-01", tz="UTC").tz_convert(dtindex.tz)
    else:
        epoch = pd.Timestamp("1970-01-01")

    return np.array((dtindex - epoch) / pd.Timedelta("1s"))


def _as_index(time):
    if isinstance(time, pd.DatetimeIndex):
        return time
    if isinstance(time, (dt.datetime, pd.Timestamp, dt.date, np.datetime64)):
        return pd.DatetimeIndex([pd.Timestamp(time)])
    return pd.DatetimeIndex(time)


def get_solarposition(time, latitude, longitude, altitude=None, pressure=None,
                      method="nrel_numpy", temperature=12.0, **kwargs):
    """
    A convenience wrapper for the solar position calculators.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Must be localized or UTC will be assumed.
    latitude : float
        Latitude in decimal degrees.
    longitude : float
        Longitude in decimal degrees.
    altitude : float, optional
        If not specified, computed from ``pressure``. Assumed to be 0 m
        if ``pressure`` is not supplied.
    pressure : float, optional
        If not specified, computed from ``altitude``. Assumed to be 101325 Pa
        if ``altitude`` is not supplied.
    method : string, default 'nrel_numpy'
        Only ``'nrel_numpy'`` is ported; it is an implementation of the NREL
        SPA algorithm described in [1] (default, recommended).
    temperature : float, default 12
        Degrees C.

    References
    ----------
    .. [1] I. Reda and A. Andreas, Solar position algorithm for solar radiation
       applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
    """
    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = pres2alt(pressure)
    elif pressure is None:
        pressure = alt2pres(altitude)

    method = method.lower()
    time = _as_index(time)
    if method != "nrel_numpy":
        raise ValueError("Invalid solar position method")
    return spa_python(time, latitude, longitude, altitude, pressure,
                      temperature, **kwargs)


def spa_python(time, latitude, longitude, altitude=0., pressure=101325.,
               temperature=12., delta_t=67.0, atmos_refract=None, how="numpy",
               numthreads=4):
    """
    Calculate the solar position using a Mojo implementation of the NREL SPA
    algorithm.

    ``how`` and ``numthreads`` are accepted for signature compatibility with
    upstream and ignored: the whole array goes through one kernel call.

    Returns
    -------
    DataFrame
        Columns: apparent_zenith, zenith, apparent_elevation, elevation,
        azimuth, equation_of_time.
    """
    lat = latitude
    lon = longitude
    elev = altitude
    pressure = pressure / 100  # pressure must be in millibars for calculation

    atmos_refract = atmos_refract or 0.5667

    time = _as_index(time)
    unixtime = _datetime_to_unixtime(time)

    if delta_t is None:
        time_utc = time.tz_convert("UTC") if time.tz is not None else time
        delta_t = spa.calculate_deltat(time_utc.year, time_utc.month)

    app_zenith, zenith, app_elevation, elevation, azimuth, eot = \
        spa.solar_position(unixtime, lat, lon, elev, pressure, temperature,
                           delta_t, atmos_refract, numthreads)

    return pd.DataFrame({
        "apparent_zenith": app_zenith,
        "zenith": zenith,
        "apparent_elevation": app_elevation,
        "elevation": elevation,
        "azimuth": azimuth,
        "equation_of_time": eot,
    }, index=time)


def nrel_earthsun_distance(time, how="numpy", delta_t=67.0, numthreads=4):
    """
    Calculates the distance from the earth to the sun using the
    NREL SPA algorithm.

    Returns
    -------
    dist : pd.Series
        Earth-sun distance in AU.
    """
    time = _as_index(time)
    unixtime = _datetime_to_unixtime(time)

    if delta_t is None:
        time_utc = time.tz_convert("UTC") if time.tz is not None else time
        delta_t = spa.calculate_deltat(time_utc.year, time_utc.month)

    dist = spa.earthsun_distance(unixtime, delta_t, numthreads)
    return pd.Series(dist, index=time)


def _calculate_simple_day_angle(dayofyear, offset=1):
    """
    Calculates the day angle for the Earth's orbit around the Sun.

    offset : int, default 1
        For the Spencer method, offset=1; for the ASCE method, offset=0
    """
    n, (doy,) = vectors(dayofyear)
    day_angle = out(n)
    lib().mpv_calculate_simple_day_angle(addr(doy), float(offset), addr(day_angle), n)
    return shaped(day_angle, dayofyear)


def _day_angle_model(symbol: str, dayofyear):
    n, (doy,) = vectors(dayofyear)
    res = out(n)
    day_angle = out(n)
    getattr(lib(), symbol)(addr(doy), addr(res), n, addr(day_angle))
    return shaped(res, dayofyear)


def equation_of_time_spencer71(dayofyear):
    """
    Equation of time from Duffie & Beckman and attributed to Spencer
    (1971) and Iqbal (1983).
    """
    return _day_angle_model("mpv_equation_of_time_spencer71", dayofyear)


def equation_of_time_pvcdrom(dayofyear):
    """
    Equation of time from PVCDROM.
    """
    return _day_angle_model("mpv_equation_of_time_pvcdrom", dayofyear)


def declination_spencer71(dayofyear):
    """
    Solar declination from Duffie & Beckman and attributed to
    Spencer (1971) and Iqbal (1983). Return units are radians, not degrees.
    """
    return _day_angle_model("mpv_declination_spencer71", dayofyear)


def declination_cooper69(dayofyear):
    """
    Solar declination from Duffie & Beckman and attributed to Cooper (1969).
    Return units are radians, not degrees.
    """
    return _day_angle_model("mpv_declination_cooper69", dayofyear)


def solar_azimuth_analytical(latitude, hourangle, declination, zenith):
    """
    Analytical expression of solar azimuth angle based on spherical
    trigonometry.
    """
    n, (lat, ha, dec, z) = vectors(latitude, hourangle, declination, zenith)
    azimuth = out(n)
    lib().mpv_solar_azimuth_analytical(
        addr(lat), addr(ha), addr(dec), addr(z), addr(azimuth), n
    )
    # Upstream's `numer` is plain `np.cos(zenith) * np.sin(latitude) - ...`, so a
    # Series latitude or declination is unwrapped by `np.sin` and the result is
    # a bare array; `np.sign(hourangle)` is a ufunc, so a Series hour angle
    # survives into the product and comes back as a Series. `zenith` is read by
    # `np.cos`/`np.sin` too, but those only ever see it inside `numer`/`denom`,
    # whose own unwrapping is already covered by latitude.
    flat = shaped(azimuth, latitude, hourangle, declination)
    if not is_series(hourangle):
        flat = np.asarray(flat)
    return flat


def solar_zenith_analytical(latitude, hourangle, declination):
    """
    Analytical expression of solar zenith angle based on spherical
    trigonometry.
    """
    n, (lat, ha, dec) = vectors(latitude, hourangle, declination)
    zenith = out(n)
    lib().mpv_solar_zenith_analytical(
        addr(lat), addr(ha), addr(dec), addr(zenith), n
    )
    return shaped(zenith, latitude, hourangle, declination)


def hour_angle(times, longitude, equation_of_time):
    """
    Hour angle in local solar time. Zero at local solar noon.
    """
    # times must be localized
    if not times.tz:
        raise ValueError("times must be localized")

    # hours - timezone = (times - normalized_times) - (naive_times - times)
    tzs = np.array([ts.utcoffset().total_seconds() for ts in times]) / 3600

    hrs_minus_tzs = _times_to_hours_after_local_midnight(times) - tzs

    n, (hrs, eot) = vectors(hrs_minus_tzs, equation_of_time)
    ha = out(n)
    lib().mpv_hour_angle(addr(hrs), float(longitude), addr(eot), addr(ha), n)
    return shaped(ha, hrs_minus_tzs, longitude, equation_of_time)


def _times_to_hours_after_local_midnight(times):
    """convert local pandas datetime indices to array of hours as floats"""
    # times must be localized
    if not times.tz:
        raise ValueError("times must be localized")

    # Some timezones have a DST shift at midnight:
    #   11:59pm -> 1:00am - results in a nonexistent midnight
    #   12:59am -> 12:00am - results in an ambiguous midnight
    # We remove the timezone before normalizing for this reason.
    naive_normalized_times = times.tz_localize(None).normalize()

    # Use Pandas functionality for shifting nonexistent times forward
    normalized_times = naive_normalized_times.tz_localize(
        times.tz, nonexistent="shift_forward", ambiguous="raise")

    hrs = (times - normalized_times) / pd.Timedelta("1h")

    # ensure array return instead of a version-dependent pandas <T>Index
    return np.array(hrs)


def _hour_angle_to_hours(times, hourangle, longitude, equation_of_time):
    """converts hour angles in degrees to hours as a numpy array"""
    # times must be localized
    if not times.tz:
        raise ValueError("times must be localized")

    tzs = np.array([ts.utcoffset().total_seconds() for ts in times]) / 3600
    hours = (hourangle - longitude - equation_of_time / 4.) / 15. + 12. + tzs
    return np.asarray(hours)


def _local_times_from_hours_since_midnight(times, hours):
    """
    converts hours since midnight from an array of floats to localized times
    """
    # times must be localized
    if not times.tz:
        raise ValueError("times must be localized")

    # normalize local times to previous local midnight and add the hours until
    # sunrise, sunset, and transit
    return times.normalize() + pd.to_timedelta(hours, unit="h")


def sun_rise_set_transit_geometric(times, latitude, longitude, declination,
                                   equation_of_time):
    """
    Geometric calculation of solar sunrise, sunset, and transit.

    .. warning:: The geometric calculation assumes a circular earth orbit with
        the sun as a point source at its center, and neglects the effect of
        atmospheric refraction on zenith. The error depends on location and
        time of year but is of order 10 minutes.
    """
    # times must be localized
    if not times.tz:
        raise ValueError("times must be localized")

    n, (dec,) = vectors(declination)
    sunset_angle = out(n)
    lib().mpv_sunset_angle(float(latitude), addr(dec), addr(sunset_angle), n)
    # solar noon is at hour angle zero
    # so sunrise is just negative of sunset
    sunrise_angle = -sunset_angle
    sunrise_hour = _hour_angle_to_hours(
        times, sunrise_angle, longitude, equation_of_time)
    sunset_hour = _hour_angle_to_hours(
        times, sunset_angle, longitude, equation_of_time)
    transit_hour = _hour_angle_to_hours(times, 0, longitude, equation_of_time)
    sunrise = _local_times_from_hours_since_midnight(times, sunrise_hour)
    sunset = _local_times_from_hours_since_midnight(times, sunset_hour)
    transit = _local_times_from_hours_since_midnight(times, transit_hour)
    return sunrise, sunset, transit
