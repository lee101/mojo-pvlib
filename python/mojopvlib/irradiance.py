"""pvlib.irradiance: irradiance decomposition and transposition models.

Same names, same argument order and same defaults as `pvlib.irradiance`; the
arithmetic lives in `src/irradiance.mojo`. The dispatch-only functions
(`get_total_irradiance`, `get_sky_diffuse`, `complete_irradiance`) stay in
Python here, exactly as they are dictionary and branch lookups upstream.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from . import atmosphere
from ._lib import (
    EXTRA_RADIATION_METHODS, addr, broadcast, dirint_coeffs, is_series, lib,
    out, perez_coefficient_table, shaped, vectors,
)
from . import tools
from . import solarposition


def _array_out(value):
    """Upstream's non-datetime output rule: a Series loses its container."""
    return value.values if isinstance(value, pd.Series) else value


def _scalar_out(value):
    """Upstream's scalar output rule: unwrap a 1-element result to a number.

    `np.isscalar` is true of a numpy scalar as well as a Python number, so a
    `np.float64` passes through unchanged and a 1-element array becomes a
    Python `float`, which is what `.item()` gives.
    """
    if np.isscalar(value):
        return value
    return np.asarray(value).item()


def _pandas_to_utc(pd_object):
    """Upstream localizes a tz-naive time before taking its day of year."""
    if isinstance(pd_object, pd.Series):
        if pd_object.dt.tz is None:
            return pd_object.dt.tz_localize("UTC")
        return pd_object.dt.tz_convert("UTC")
    if pd_object.tz is None:
        return pd_object.tz_localize("UTC")
    return pd_object.tz_convert("UTC")


def _pandas_to_doy(pd_object):
    return _pandas_to_utc(pd_object).dayofyear


def _handle_extra_radiation_types(datetime_or_doy, epoch_year):
    """Upstream's input triage for `get_extra_radiation`, transcribed.

    It returns the three callables upstream builds: one to get a day of year,
    one to get a `DatetimeIndex`, and one to give the answer the right
    container. The output rule is the part that is easy to lose -- a
    `DatetimeIndex` gives a Series, a scalar datetime gives a bare number, and
    everything else gives a bare ndarray with any Series unwrapped -- and it
    applies to the `asce` and `spencer` methods just as much as to `nrel`.
    """
    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        def to_doy(x):
            return _pandas_to_doy(x)

        def to_datetimeindex(x):
            return x

        def to_output(value):
            return pd.Series(value, index=datetime_or_doy)

        return to_doy, to_datetimeindex, to_output
    if isinstance(datetime_or_doy, pd.Timestamp):
        return (_pandas_to_doy,
                _datetimelike_scalar_to_datetimeindex,
                _scalar_out)
    if isinstance(datetime_or_doy, (datetime.date, datetime.datetime,
                                   np.datetime64)):
        return (_datetimelike_scalar_to_doy,
                _datetimelike_scalar_to_datetimeindex,
                _scalar_out)
    if np.isscalar(datetime_or_doy):  # ints and floats of various types
        return (lambda x: x,
                lambda x: _doy_to_datetimeindex(x, epoch_year),
                _scalar_out)
    # assume we have an array-like object of doy
    def _to_output(value):
        # upstream's `_array_out` unwraps a Series, and a pd.Index of day
        # numbers survives the day-angle ufunc as a pd.Index
        if isinstance(value, pd.Series):
            return value.values
        return value

    return (lambda x: x,
            lambda x: _doy_to_datetimeindex(x, epoch_year),
            _to_output)


def _datetimelike_scalar_to_datetimeindex(time):
    return pd.DatetimeIndex([pd.Timestamp(time)])


def _datetimelike_scalar_to_doy(time):
    """Upstream's day-of-year for a scalar datetime: a 1-element `Index`."""
    return _pandas_to_doy(_datetimelike_scalar_to_datetimeindex(time))


def _doy_to_datetimeindex(doy, epoch_year):
    return pd.DatetimeIndex([
        pd.Timestamp(f"{int(epoch_year)}-01-01") + pd.Timedelta(days=int(d) - 1)
        for d in np.atleast_1d(doy)
    ])


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


def _doy(datetime_or_doy, epoch_year=2014):
    """Day of year for the input, whatever kind of time-like it is."""
    to_doy, _, _ = _handle_extra_radiation_types(datetime_or_doy, epoch_year)
    return to_doy(datetime_or_doy)


def _doy_index(datetime_or_doy, epoch_year=2014):
    """A DatetimeIndex for the input, whatever kind of time-like it is."""
    _, to_datetimeindex, _ = _handle_extra_radiation_types(
        datetime_or_doy, epoch_year)
    return to_datetimeindex(datetime_or_doy)


def _indexed_out(data, datetime_or_doy):
    """Return a DataFrame indexed by `times` when the input was datetimes."""
    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        return pd.DataFrame(data, index=datetime_or_doy)
    return data


def get_extra_radiation(datetime_or_doy, solar_constant=1366.1, method="spencer",
                        epoch_year=2014, **kwargs):
    """
    Determine extraterrestrial radiation from day of year.

    Parameters
    ----------
    datetime_or_doy : numeric, array, date, datetime, Timestamp, DatetimeIndex
        Day of year, array of days of year, or datetime-like object
    solar_constant : float, default 1366.1
        The solar constant. [Wm⁻²]
    method : string, default `spencer`
        The method by which the extraterrestrial radiation should be
        calculated. Options include: `asce`, `spencer`, `nrel`.
    epoch_year : int, default 2014
        The year in which a day of year input will be calculated. Only
        applies to the `nrel` method.

    Returns
    -------
    dni_extra : float, array, or Series
        The extraterrestrial radiation normal to the sun. [Wm⁻²]
    """
    # `asce` and `spencer` are driven by `_calculate_simple_day_angle`, a
    # numpy ufunc, so their answer carries the container of the day angle that
    # came out of it: a pd.Index in gives a pd.Index out and a Series in gives a
    # Series. The `nrel` method goes through `_array_out` upstream, which strips
    # the Series, so the two paths disagree on a Series input and agree
    # otherwise. Reproducing the dispatch rather than the output rule is what
    # keeps both true.
    to_doy, to_datetimeindex, to_output = _handle_extra_radiation_types(
        datetime_or_doy, epoch_year)
    method = method.lower()
    if method in EXTRA_RADIATION_METHODS:
        doy = to_doy(datetime_or_doy)
        n, (d,) = vectors(doy)
        ea = out(n)
        lib().mpv_get_extra_radiation(
            addr(d), addr(ea), n, float(solar_constant), EXTRA_RADIATION_METHODS[method]
        )
        # upstream's container rule, keyed on the *input* rather than on the
        # day angle it derived, because `_scalar_out`/`_array_out`/`to_series`
        # are what actually decide it. Reproducing the triage is what keeps
        # all three paths (DatetimeIndex / scalar datetime / day-of-year
        # array) correct at once, including the 2-D case the flat ABI has to
        # reshape on the way out.
        return to_output(shaped(ea, doy))
    elif method == "nrel":
        # upstream hands the *original* input to `to_datetimeindex`, not the
        # day of year it derived: for a DatetimeIndex that is a no-op, and for
        # a day-of-year series it builds one day per element. Passing the day
        # of year instead would shift a DatetimeIndex's timestamps by a day.
        times = to_datetimeindex(datetime_or_doy)
        RoverR0sqrd = solarposition.nrel_earthsun_distance(times, **kwargs) ** (-2)
    else:
        raise ValueError("Invalid method: %s" % method)

    Ea = solar_constant * RoverR0sqrd
    # upstream's `to_output` does the final container choice, which is keyed
    # on the input rather than on the day angle the closed-form methods used
    return to_output(Ea)


def aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    Calculates the dot product of the sun position unit vector and the surface

    Usage note: When the sun is behind the surface the value returned is
    negative.  For many uses negative values must be set to zero.
    """
    n, (t, az, z, saz) = vectors(surface_tilt, surface_azimuth, solar_zenith,
                                 solar_azimuth)
    projection = out(n)
    lib().mpv_aoi_projection(
        addr(t), addr(az), addr(z), addr(saz), addr(projection), n
    )
    return shaped(projection, surface_tilt, surface_azimuth, solar_zenith,
                  solar_azimuth)


def aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    Calculates the angle of incidence of the solar vector on a surface.
    This is the angle between the solar vector and the surface normal.
    """
    n, (t, az, z, saz) = vectors(surface_tilt, surface_azimuth, solar_zenith,
                                 solar_azimuth)
    angle = out(n)
    lib().mpv_aoi(addr(t), addr(az), addr(z), addr(saz), addr(angle), n)
    return shaped(angle, surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)


def beam_component(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni):
    """
    Calculates the beam component of the plane of array irradiance.
    """
    n, (t, az, z, saz, d) = vectors(surface_tilt, surface_azimuth, solar_zenith,
                                    solar_azimuth, dni)
    beam = out(n)
    lib().mpv_beam_component(
        addr(t), addr(az), addr(z), addr(saz), addr(d), addr(beam), n
    )
    return shaped(beam, surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
                  dni)


def get_total_irradiance(surface_tilt, surface_azimuth, solar_zenith,
                         solar_azimuth, dni, ghi, dhi, dni_extra=None,
                         airmass=None, albedo=0.25, surface_type=None,
                         model="isotropic", model_perez="allsitescomposite1990",
                         diffuse_components=False):
    r"""
    Determine the total irradiance on a surface.

    Returns
    -------
    total_irrad : dict
        Contains keys ``'poa_global', 'poa_direct', 'poa_diffuse',
        'poa_sky_diffuse', 'poa_ground_diffuse'``. [Wm⁻²]
    """
    poa_sky_diffuse = get_sky_diffuse(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi,
        dni_extra=dni_extra, airmass=airmass, model=model, model_perez=model_perez,
        return_components=diffuse_components)

    poa_ground_diffuse = get_ground_diffuse(surface_tilt, ghi, albedo, surface_type)
    aoi_ = aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    irrads = poa_components(aoi_, dni, poa_sky_diffuse, poa_ground_diffuse)
    return irrads


def get_sky_diffuse(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
                    dni, ghi, dhi, dni_extra=None, airmass=None, model="isotropic",
                    model_perez="allsitescomposite1990", return_components=False):
    r"""
    Determine in-plane sky diffuse irradiance component using the specified
    sky diffuse irradiance model.
    """
    model = model.lower()

    if return_components and model in {"klucher", "king"}:
        raise ValueError("return_components is not supported for"
                         f" model {model}")

    if dni_extra is None and model in {"haydavies", "reindl", "perez",
                                       "perez-driesse"}:
        raise ValueError(f"dni_extra is required for model {model}")

    if model == "isotropic":
        sky = isotropic(surface_tilt, dhi, return_components=return_components)
    elif model == "klucher":
        sky = klucher(surface_tilt, surface_azimuth, dhi, ghi, solar_zenith,
                      solar_azimuth)
    elif model == "haydavies":
        sky = haydavies(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                        solar_zenith, solar_azimuth,
                        return_components=return_components)
    elif model == "reindl":
        sky = reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
                     solar_zenith, solar_azimuth,
                     return_components=return_components)
    elif model == "king":
        sky = king(surface_tilt, dhi, ghi, solar_zenith)
    elif model == "perez":
        if airmass is None:
            airmass = atmosphere.get_relative_airmass(solar_zenith)
        sky = perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                    solar_zenith, solar_azimuth, airmass, model=model_perez,
                    return_components=return_components)
    else:
        raise ValueError(f"invalid model selection {model}")

    return sky


def poa_components(aoi, dni, poa_sky_diffuse, poa_ground_diffuse):
    r'''
    Determine the total plane of array irradiance.

    Returns
    -------
    irrads : dict
        Contains keys ``'poa_global', 'poa_direct', 'poa_diffuse',
        'poa_sky_diffuse', 'poa_ground_diffuse'`` plus any diffuse components.
    '''
    if isinstance(poa_sky_diffuse, dict):
        sky_components = poa_sky_diffuse.copy()
        total_poa_sky_diffuse = sky_components.pop('poa_sky_diffuse')
    elif isinstance(poa_sky_diffuse, pd.DataFrame):
        sky_components = poa_sky_diffuse.to_dict(orient='series')
        total_poa_sky_diffuse = sky_components.pop('poa_sky_diffuse')
    else:
        sky_components = {}
        total_poa_sky_diffuse = poa_sky_diffuse

    # poa_direct, poa_diffuse and poa_global are elementwise functions of the
    # same four buffers, so one kernel pass writes all three instead of three
    # whole-array numpy expressions each re-reading them.
    n, (a, d, sky, grd) = vectors(aoi, dni, total_poa_sky_diffuse,
                                   poa_ground_diffuse)
    poa_direct = out(n)
    poa_diffuse = out(n)
    poa_global = out(n)
    lib().mpv_poa_components(
        addr(a), addr(d), addr(sky), addr(grd),
        addr(poa_direct), addr(poa_diffuse), addr(poa_global), n,
    )

    # `poa_ground_diffuse` is returned exactly as `get_ground_diffuse` produced
    # it -- upstream returns the scalar here too -- but it still takes part in
    # the broadcasting that decides the shape of the three totals.
    irrads = {
        'poa_global': shaped(poa_global, aoi, dni, total_poa_sky_diffuse),
        'poa_direct': shaped(poa_direct, aoi, dni, total_poa_sky_diffuse),
        'poa_diffuse': shaped(poa_diffuse, aoi, dni, total_poa_sky_diffuse),
        'poa_sky_diffuse': total_poa_sky_diffuse,
        'poa_ground_diffuse': poa_ground_diffuse,
        **sky_components
    }

    # Upstream decides the container from `poa_direct`, which is
    # `max(dni * cos(radians(aoi)), 0)`: a Series survives that product if
    # either factor is one. Testing the operands rather than the product keeps
    # the check honest for a zero-length or scalar-only call, where there is no
    # `poa_direct` to inspect.
    if is_series(dni) or is_series(aoi):
        irrads = pd.DataFrame(irrads)

    return irrads


def get_ground_diffuse(surface_tilt, ghi, albedo=.25, surface_type=None):
    r'''
    Estimate diffuse irradiance on a tilted surface from ground reflections.

    Ground diffuse irradiance is calculated as

    .. math::

       G_{ground} = GHI \times \rho \times \frac{1 - \cos\beta}{2}
    '''
    if surface_type is not None:
        albedo = _SURFACE_ALBEDOS[surface_type]

    n, (t, g) = vectors(surface_tilt, ghi)
    ground = out(n)
    lib().mpv_get_ground_diffuse(addr(t), addr(g), addr(ground), n, float(albedo))
    return shaped(ground, surface_tilt, ghi)


_SURFACE_ALBEDOS = {
    'urban': 0.18, 'grass': 0.20, 'fresh grass': 0.26, 'snow': 0.65,
    'fresh snow': 0.75, 'asphalt': 0.12, 'concrete': 0.30, 'aluminum': 0.85,
    'copper': 0.74, 'fresh steel': 0.35, 'dirty steel': 0.08, 'sea': 0.06,
    'sand': 0.40, 'soil': 0.17,
}


def isotropic(surface_tilt, dhi, return_components=False):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using
    the isotropic sky model.

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2}
    '''
    n, (t, d) = vectors(surface_tilt, dhi)
    sky = out(n)
    poa_isotropic = out(n)
    lib().mpv_isotropic(addr(t), addr(d), addr(sky), addr(poa_isotropic), n)
    sky = shaped(sky, surface_tilt, dhi)

    if return_components:
        return _components(
            {'poa_sky_diffuse': sky,
             'poa_isotropic': shaped(poa_isotropic, surface_tilt, dhi)}, sky)
    return sky


def _components(diffuse_components, sky):
    """Upstream wraps the component dict in a DataFrame when the sky value is
    a Series, taking its index from the components themselves. `sky` is the
    already-shaped `poa_sky_diffuse`, which is exactly the test upstream makes.
    """
    if is_series(sky):
        return pd.DataFrame(diffuse_components)
    return diffuse_components


def klucher(surface_tilt, surface_azimuth, dhi, ghi, solar_zenith, solar_azimuth):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using the
    Klucher (1979) model.
    '''
    n, (t, az, d, g, z, saz) = vectors(surface_tilt, surface_azimuth, dhi, ghi,
                                      solar_zenith, solar_azimuth)
    sky = out(n)
    lib().mpv_klucher(addr(t), addr(az), addr(d), addr(g), addr(z), addr(saz),
                      addr(sky), n)
    return shaped(sky, surface_tilt, surface_azimuth, dhi, ghi, solar_zenith,
                  solar_azimuth)


def haydavies(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
              solar_zenith=None, solar_azimuth=None, projection_ratio=None,
              return_components=False):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using the
    Hay-Davies (1980) model.
    '''
    use_projection_ratio = projection_ratio is not None
    n, (t, az, d, dn, ex, z, saz, pr) = vectors(
        surface_tilt, surface_azimuth, dhi, dni, dni_extra,
        solar_zenith if solar_zenith is not None else 0.0,
        solar_azimuth if solar_azimuth is not None else 0.0,
        projection_ratio if use_projection_ratio else 0.0)
    sky = out(n)
    poa_isotropic = out(n)
    poa_circumsolar = out(n)
    lib().mpv_haydavies(
        addr(t), addr(az), addr(d), addr(dn), addr(ex), addr(z), addr(saz),
        addr(pr), addr(sky), addr(poa_isotropic), addr(poa_circumsolar), n,
        int(use_projection_ratio)
    )
    sky = shaped(sky, surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                 solar_zenith, solar_azimuth)
    if return_components:
        # upstream keeps every component's container decided by `dhi`, which is
        # the one input they are all multiplied by.
        return _components({
            'poa_sky_diffuse': sky,
            'poa_isotropic': shaped(poa_isotropic, dhi),
            'poa_circumsolar': shaped(poa_circumsolar, dhi),
        }, sky)
    return sky


def reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
           solar_zenith, solar_azimuth, return_components=False):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using the
    Reindl (1990) model.
    '''
    n, (t, az, d, dn, g, ex, z, saz) = vectors(
        surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra, solar_zenith,
        solar_azimuth)
    sky = out(n)
    poa_isotropic = out(n)
    poa_circumsolar = out(n)
    poa_horizon = out(n)
    lib().mpv_reindl(
        addr(t), addr(az), addr(d), addr(dn), addr(g), addr(ex), addr(z), addr(saz),
        addr(sky), addr(poa_isotropic), addr(poa_circumsolar), addr(poa_horizon), n
    )
    sky = shaped(sky, surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
                 solar_zenith, solar_azimuth)
    if return_components:
        return _components({
            'poa_sky_diffuse': sky,
            'poa_isotropic': shaped(poa_isotropic, dhi),
            'poa_circumsolar': shaped(poa_circumsolar, dhi),
            'poa_horizon': shaped(poa_horizon, dhi),
        }, sky)
    return sky


def king(surface_tilt, dhi, ghi, solar_zenith):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using the
    King model.
    '''
    n, (t, d, g, z) = vectors(surface_tilt, dhi, ghi, solar_zenith)
    sky = out(n)
    lib().mpv_king(addr(t), addr(d), addr(g), addr(z), addr(sky), n)
    return shaped(sky, surface_tilt, dhi, ghi, solar_zenith)


def perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra, solar_zenith,
          solar_azimuth, airmass, model="allsitescomposite1990",
          return_components=False):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using the
    Perez model.
    '''
    F1coeffs, F2coeffs = perez_coefficient_table(model)
    n, (t, az, d, dn, ex, z, saz, am) = vectors(
        surface_tilt, surface_azimuth, dhi, dni, dni_extra, solar_zenith,
        solar_azimuth, airmass)
    sky = out(n)
    poa_isotropic = out(n)
    poa_circumsolar = out(n)
    poa_horizon = out(n)
    lib().mpv_perez(
        addr(t), addr(az), addr(d), addr(dn), addr(ex), addr(z), addr(saz),
        addr(am), addr(F1coeffs), addr(F2coeffs), addr(sky), addr(poa_isotropic),
        addr(poa_circumsolar), addr(poa_horizon), n
    )
    sky = shaped(sky, surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                 solar_zenith, solar_azimuth, airmass)
    if return_components:
        components = {
            'poa_sky_diffuse': sky,
            'poa_isotropic': shaped(poa_isotropic, dhi),
            'poa_circumsolar': shaped(poa_circumsolar, dhi),
            'poa_horizon': shaped(poa_horizon, dhi),
        }
        # Set values of components to 0 when poa_sky_diffuse is 0
        if isinstance(sky, pd.Series):
            mask = sky == 0
            frame = pd.DataFrame(components, index=sky.index)
            frame.loc[mask] = 0
            return frame
        return {k: np.where(np.asarray(sky) == 0, 0, v) for k, v in components.items()}
    return sky


def clearsky_index(ghi, ghi_clear, max_clearsky_index=2.0):
    """
    Calculate the clearsky index.
    """
    n, (g, gc) = vectors(ghi, ghi_clear)
    k = out(n)
    lib().mpv_clearsky_index(addr(g), addr(gc), addr(k), n, float(max_clearsky_index))
    return shaped(k, ghi, ghi_clear)


def clearness_index(ghi, solar_zenith, extra_radiation, min_cos_zenith=0.065,
                    max_clearness_index=2.0):
    """
    Calculate the clearness index.
    """
    n, (g, z, ex) = vectors(ghi, solar_zenith, extra_radiation)
    kt = out(n)
    lib().mpv_clearness_index(addr(g), addr(z), addr(ex), addr(kt), n,
                              float(min_cos_zenith), float(max_clearness_index))
    return shaped(kt, ghi, solar_zenith, extra_radiation)


def clearness_index_zenith_independent(clearness_index, airmass,
                                       max_clearness_index=2.0):
    """
    Calculate the zenith angle independent clearness index.
    """
    n, (ci, am) = vectors(clearness_index, airmass)
    ktp = out(n)
    lib().mpv_clearness_index_zenith_independent(
        addr(ci), addr(am), addr(ktp), n, float(max_clearness_index)
    )
    return shaped(ktp, clearness_index, airmass)



def disc(ghi, solar_zenith, datetime_or_doy, pressure=101325, min_cos_zenith=0.065,
         max_zenith=87, max_airmass=12):
    """
    Determine DNI from GHI using the DISC model.
    """
    use_pressure = pressure is not None
    n, (g, z) = vectors(ghi, solar_zenith)
    doy = broadcast(_doy(datetime_or_doy), n)
    dni = out(n)
    kt = out(n)
    airmass = out(n)
    lib().mpv_disc(
        addr(g), addr(z), addr(doy), addr(dni), addr(kt), addr(airmass), n,
        float(pressure) if use_pressure else 0.0, float(min_cos_zenith),
        float(max_zenith), float(max_airmass), int(use_pressure)
    )
    output = {'dni': shaped(dni, ghi, solar_zenith),
              'kt': shaped(kt, ghi, solar_zenith),
              'airmass': shaped(airmass, ghi, solar_zenith)}
    return _indexed_out(output, datetime_or_doy)




def dirint(ghi, solar_zenith, times, pressure=101325., use_delta_kt_prime=True,
           temp_dew=None, min_cos_zenith=0.065, max_zenith=87):
    """
    Determine DNI from GHI using the DIRINT model.
    """
    use_pressure = pressure is not None
    n, (g, z, td) = vectors(ghi, solar_zenith,
                            temp_dew if temp_dew is not None else 0.0)
    doy = broadcast(_doy(times), n)
    delta = out(n)
    scratch = np.zeros(3 * n, dtype=np.float64)
    dni = out(n)
    lib().mpv_dirint(
        addr(g), addr(z), addr(doy), addr(delta), addr(td),
        addr(dirint_coeffs()), addr(dni), n,
        float(pressure) if use_pressure else 0.0, 12.0, int(use_pressure),
        int(use_delta_kt_prime), int(temp_dew is not None),
        float(min_cos_zenith), float(max_zenith), addr(scratch)
    )
    return shaped(dni, ghi, solar_zenith)


def dirindex(ghi, ghi_clear, dni_clear, zenith, times, pressure=101325.,
             use_delta_kt_prime=True, temp_dew=None, min_cos_zenith=0.065,
             max_zenith=87):
    """
    Determine DNI from GHI using the DIRINDEX model.
    """
    dni_dirint = dirint(ghi, zenith, times, pressure=pressure,
                        use_delta_kt_prime=use_delta_kt_prime,
                        temp_dew=temp_dew, min_cos_zenith=min_cos_zenith,
                        max_zenith=max_zenith)

    dni_dirint_clearsky = dirint(ghi_clear, zenith, times, pressure=pressure,
                                 use_delta_kt_prime=use_delta_kt_prime,
                                 temp_dew=temp_dew,
                                 min_cos_zenith=min_cos_zenith,
                                 max_zenith=max_zenith)

    n, (dc, di, dcs) = vectors(dni_clear, dni_dirint, dni_dirint_clearsky)
    dni_dirindex = out(n)
    lib().mpv_dirindex(addr(dc), addr(di), addr(dcs), addr(dni_dirindex), n)
    return shaped(dni_dirindex, ghi, ghi_clear, dni_clear, zenith)


def _decomposition(symbol, ghi, zenith, dni_extra, min_cos_zenith, max_zenith,
                   datetime_or_doy, *extra_args):
    n, (g, z, ex) = vectors(ghi, zenith, dni_extra)
    dni = out(n)
    dhi = out(n)
    kt = out(n)
    getattr(lib(), symbol)(
        addr(g), addr(z), addr(ex), addr(dni), addr(dhi), addr(kt), n,
        *extra_args
    )
    data = {'dni': shaped(dni, ghi, zenith), 'dhi': shaped(dhi, ghi, zenith),
            'kt': shaped(kt, ghi, zenith)}
    return _indexed_out(data, datetime_or_doy)


def erbs(ghi, zenith, datetime_or_doy, min_cos_zenith=0.065, max_zenith=87):
    r"""
    Estimate DNI and DHI from GHI using the Erbs model.
    """
    dni_extra = get_extra_radiation(datetime_or_doy)
    return _decomposition("mpv_erbs", ghi, zenith, dni_extra, min_cos_zenith,
                          max_zenith, datetime_or_doy, float(min_cos_zenith),
                          float(max_zenith))


def orgill_hollands(ghi, zenith, datetime_or_doy, dni_extra=None,
                    min_cos_zenith=0.065, max_zenith=87):
    """Estimate DNI and DHI from GHI using the Orgill and Hollands model."""
    if dni_extra is None:
        dni_extra = get_extra_radiation(datetime_or_doy)
    return _decomposition("mpv_orgill_hollands", ghi, zenith, dni_extra,
                          min_cos_zenith, max_zenith, datetime_or_doy,
                          float(min_cos_zenith), float(max_zenith))


def boland(ghi, solar_zenith, datetime_or_doy, a_coeff=8.645, b_coeff=0.613,
           min_cos_zenith=0.065, max_zenith=87):
    r"""
    Estimate DNI and DHI from GHI using the Boland clearness index model.
    """
    dni_extra = get_extra_radiation(datetime_or_doy)
    return _decomposition("mpv_boland", ghi, solar_zenith, dni_extra,
                          min_cos_zenith, max_zenith, datetime_or_doy,
                          float(a_coeff), float(b_coeff), float(min_cos_zenith),
                          float(max_zenith))


def dni(ghi, dhi, zenith, dni_clear=None, clearsky_tolerance=1.1,
        zenith_threshold_for_zero_dni=88.0,
        zenith_threshold_for_clearsky_limit=80.0):
    """
    Calculate DNI from GHI and DHI.
    """
    use_dni_clear = dni_clear is not None
    n, (g, d, z, dc) = vectors(ghi, dhi, zenith,
                               dni_clear if use_dni_clear else 0.0)
    res = out(n)
    lib().mpv_dni(
        addr(g), addr(d), addr(z), addr(dc), addr(res), n,
        float(clearsky_tolerance), int(use_dni_clear),
        float(zenith_threshold_for_zero_dni), float(zenith_threshold_for_clearsky_limit)
    )
    return shaped(res, ghi, dhi, zenith, dni_clear)


def complete_irradiance(solar_zenith, ghi=None, dhi=None, dni=None,
                        dni_clear=None):
    r"""
    Use the component sum equations to calculate the missing series, using
    the other available time series. One of the three parameters (ghi, dhi,
    dni) must be `None`.
    """
    if ghi is not None and dhi is not None and dni is None:
        dni = irradiance_dni(ghi, dhi, solar_zenith, dni_clear=dni_clear,
                             clearsky_tolerance=1.1)
    elif dni is not None and dhi is not None and ghi is None:
        ghi = (dhi + dni * tools.cosd(solar_zenith))
    elif dni is not None and ghi is not None and dhi is None:
        dhi = (ghi - dni * tools.cosd(solar_zenith))
    else:
        raise ValueError(
            "Please check that exactly one of ghi, dhi and dni parameters "
            "is set to None"
        )
    given = [a for a in (ghi, dhi, dni) if a is not None]
    if any(np.ndim(a) > 1 for a in given):
        # upstream builds a DataFrame from the three series, and pandas only
        # accepts 1-D columns, so both sides raise the same thing here.
        raise ValueError("Per-column arrays must each be 1-dimensional")
    # Merge the outputs into a master dataframe containing 'ghi', 'dhi',
    # and 'dni' columns
    component_sum_df = pd.DataFrame({'ghi': ghi, 'dhi': dhi, 'dni': dni})
    return component_sum_df


# `dni` is the module-level name for the DNI decomposition; complete_irradiance
# calls it through this alias so the local `dni` parameter does not shadow it.
irradiance_dni = dni


def louche(ghi, solar_zenith, datetime_or_doy, max_zenith=90):
    """
    Determine DNI and DHI from GHI using the Louche model.
    """
    I0 = get_extra_radiation(datetime_or_doy)
    n, (g, z, ex) = vectors(ghi, solar_zenith, I0)
    dni = out(n)
    dhi = out(n)
    kt = out(n)
    lib().mpv_louche(addr(g), addr(z), addr(ex), addr(dni), addr(dhi), addr(kt),
                     n, float(max_zenith))
    data = {'dni': shaped(dni, ghi, solar_zenith),
            'dhi': shaped(dhi, ghi, solar_zenith),
            'kt': shaped(kt, ghi, solar_zenith)}
    return _indexed_out(data, datetime_or_doy)


def campbell_norman(zenith, transmittance, pressure=101325.0, dni_extra=1367.0):
    """
    Determine DNI, DHI, GHI from extraterrestrial flux, transmittance,
    and atmospheric pressure.

    Returns a dict of `ghi`, `dni` and `dhi`, or a DataFrame of the three
    when `zenith` is a Series. [Wm⁻²]
    """
    n, (z,) = vectors(zenith)
    ghi = out(n)
    dni = out(n)
    dhi = out(n)
    # `transmittance`, `pressure` and `dni_extra` are kernel scalars rather than
    # per-element buffers. An array argument is a length-1 operand or nothing,
    # which is what numpy's scalar arithmetic reads; anything else is rejected
    # rather than having its first element taken on the caller's behalf.
    lib().mpv_campbell_norman(
        addr(z), addr(ghi), addr(dni), addr(dhi), n,
        _kernel_scalar(transmittance),
        _kernel_scalar(pressure),
        _kernel_scalar(dni_extra),
    )
    irrads = {
        'ghi': shaped(ghi, zenith),
        'dni': shaped(dni, zenith),
        'dhi': shaped(dhi, zenith),
    }
    if isinstance(zenith, pd.Series):
        irrads = pd.DataFrame(irrads)
    return irrads


def diffuse_par_spitters(daily_solar_zenith, global_diffuse_fraction):
    r"""
    Derive daily diffuse fraction of Photosynthetically Active Radiation (PAR)
    from daily average solar zenith and diffuse fraction of daily insolation.

    The relationship is based on the work of Spitters et al. (1986). [unitless]
    """
    n, (zen, kd) = vectors(daily_solar_zenith, global_diffuse_fraction)
    par_diffuse_fraction = out(n)
    lib().mpv_diffuse_par_spitters(
        addr(zen), addr(kd), addr(par_diffuse_fraction), n
    )
    return shaped(par_diffuse_fraction, daily_solar_zenith,
                  global_diffuse_fraction)
