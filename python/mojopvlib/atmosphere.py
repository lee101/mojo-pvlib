"""pvlib.atmosphere: airmass and the clear-sky turbidity helpers.

Same names, same argument order and same defaults as `pvlib.atmosphere`; the
arithmetic lives in `src/atmosphere.mojo`.
"""

from __future__ import annotations

import numpy as np

from ._lib import AIRMASS_MODELS, addr, lib, out, shaped, vectors


def pres2alt(pressure):
    '''
    Determine altitude from site pressure.

    Parameters
    ----------
    pressure : numeric
        Atmospheric pressure. [Pa]

    Returns
    -------
    altitude : numeric
        Altitude above sea level. [m]
    '''
    n, (p,) = vectors(pressure)
    alt = out(n)
    lib().mpv_pres2alt(addr(p), addr(alt), n)
    return shaped(alt, pressure)


def alt2pres(altitude):
    '''
    Determine site pressure from altitude.

    Parameters
    ----------
    altitude : numeric
        Altitude above sea level. [m]

    Returns
    -------
    pressure : numeric
        Atmospheric pressure. [Pa]
    '''
    n, (a,) = vectors(altitude)
    press = out(n)
    lib().mpv_alt2pres(addr(a), addr(press), n)
    return shaped(press, altitude)


def get_absolute_airmass(airmass_relative, pressure=101325.):
    r'''
    Determine absolute (pressure-adjusted) airmass from relative
    airmass and pressure.

    The calculation for absolute airmass (:math:`AM_a`) is

    .. math::
        AM_a = AM_r \frac{P}{101325}
    '''
    n, (am, pres) = vectors(airmass_relative, pressure)
    absolute = out(n)
    lib().mpv_get_absolute_airmass(addr(am), addr(pres), addr(absolute), n)
    return shaped(absolute, airmass_relative, pressure)


def get_relative_airmass(zenith, model="kastenyoung1989"):
    '''
    Calculate relative (not pressure-adjusted) airmass at sea level.

    Parameter ``model`` allows selection of different airmass models:
    'simple', 'kasten1966', 'youngirvine1967', 'kastenyoung1989' (default),
    'gueymard1993', 'young1994', 'pickering2002', 'gueymard2003'.

    Returns NaN values for any zenith angle greater than 90 degrees.
    '''
    n, (z,) = vectors(zenith)
    airmass = out(n)
    model = model.lower()
    if model not in AIRMASS_MODELS:
        raise ValueError(f"{model} is not a valid model for relativeairmass")
    lib().mpv_get_relative_airmass(addr(z), addr(airmass), n, AIRMASS_MODELS[model])
    return shaped(airmass, zenith)


def gueymard94_pw(temp_air, relative_humidity):
    r"""
    Calculates precipitable water (cm) from ambient air temperature (C)
    and relatively humidity (%) using an empirical model. The accuracy of
    this method is approximately 20% for moderate PW (1-3 cm) and less
    accurate otherwise.
    """
    n, (t, rh) = vectors(temp_air, relative_humidity)
    pw = out(n)
    lib().mpv_gueymard94_pw(addr(t), addr(rh), addr(pw), n)
    return shaped(pw, temp_air, relative_humidity)


def bird_hulstrom80_aod_bb(aod380, aod500):
    """
    Approximate broadband aerosol optical depth.

    Bird and Hulstrom developed a correlation for broadband aerosol optical
    depth (AOD) using two wavelengths, 380 nm and 500 nm.
    """
    n, (a380, a500) = vectors(aod380, aod500)
    aod_bb = out(n)
    lib().mpv_bird_hulstrom80_aod_bb(addr(a380), addr(a500), addr(aod_bb), n)
    return shaped(aod_bb, aod380, aod500)


def kasten96_lt(airmass_absolute, precipitable_water, aod_bb):
    """
    Calculate Linke turbidity  using Kasten pyrheliometric formula.

    Note that broadband aerosol optical depth (AOD) can be approximated by AOD
    measured at 700 nm according to Molineaux. Bird and Hulstrom offer an
    alternate approximation using AOD measured at 380 nm and 500 nm.
    """
    n, (am, pw, aod) = vectors(airmass_absolute, precipitable_water, aod_bb)
    lt = out(n)
    lib().mpv_kasten96_lt(addr(am), addr(pw), addr(aod), addr(lt), n)
    return shaped(lt, airmass_absolute, precipitable_water, aod_bb)


def rh_from_tdew(temp_air, temp_dew, coeff=(6.112, 17.62, 243.12)):
    """
    Calculate relative humidity from dewpoint temperature using the Magnus
    equation.

    ``coeff`` is the Magnus triple (A, B, C); the default is the
    WMO-recommended form for saturation over liquid water, valid from -45 to
    +60 degC.
    """
    n, (ta, td) = vectors(temp_air, temp_dew)
    rh = out(n)
    c0, c1, c2 = coeff
    lib().mpv_rh_from_tdew(
        addr(ta), addr(td), addr(rh), n, float(c0), float(c1), float(c2)
    )
    return shaped(rh, temp_air, temp_dew)


def tdew_from_rh(temp_air, relative_humidity, coeff=(6.112, 17.62, 243.12)):
    """
    Calculate dewpoint temperature using the Magnus equation.

    This is a reversal of the calculation in :py:func:`rh_from_tdew`.
    """
    n, (ta, rh) = vectors(temp_air, relative_humidity)
    dewpoint = out(n)
    c0, c1, c2 = coeff
    lib().mpv_tdew_from_rh(
        addr(ta), addr(rh), addr(dewpoint), n, float(c0), float(c1), float(c2)
    )
    return shaped(dewpoint, temp_air, relative_humidity)


def angstrom_aod_at_lambda(aod0, lambda0, alpha=1.14, lambda1=700.0):
    """
    Get AOD at specified wavelength using Angstrom turbidity model.
    """
    n, (a0, l0) = vectors(aod0, lambda0)
    aod1 = out(n)
    lib().mpv_angstrom_aod_at_lambda(
        addr(a0), addr(l0), addr(aod1), n, float(np.ravel(alpha)[0]),
        float(np.ravel(lambda1)[0])
    )
    return shaped(aod1, aod0, lambda0)


def angstrom_alpha(aod1, lambda1, aod2, lambda2):
    r"""
    Calculate Angstrom alpha exponent.
    """
    n, (a1, l1, a2, l2) = vectors(aod1, lambda1, aod2, lambda2)
    alpha = out(n)
    lib().mpv_angstrom_alpha(
        addr(a1), addr(l1), addr(a2), addr(l2), addr(alpha), n
    )
    return shaped(alpha, aod1, lambda1, aod2, lambda2)


HELLMANN_SURFACE_EXPONENTS = {
    'unstable_air_above_open_water_surface': 0.06,
    'neutral_air_above_open_water_surface': 0.10,
    'stable_air_above_open_water_surface': 0.27,
    'unstable_air_above_flat_open_coast': 0.11,
    'neutral_air_above_flat_open_coast': 0.16,
    'stable_air_above_flat_open_coast': 0.40,
    'unstable_air_above_human_inhabited_areas': 0.27,
    'neutral_air_above_human_inhabited_areas': 0.34,
    'stable_air_above_human_inhabited_areas': 0.60,
}


def windspeed_powerlaw(wind_speed_reference, height_reference,
                       height_desired, exponent=None, surface_type=None):
    r"""
    Estimate wind speed for different heights.

    The model is based on the power law equation by Hellmann, with the
    exponent either given directly or looked up in
    :data:`HELLMANN_SURFACE_EXPONENTS` from ``surface_type``. Returns NaN when
    the reference heights or wind speed are negative.
    """
    if surface_type is not None and exponent is None:
        # use the Hellmann exponent from dictionary
        exponent = HELLMANN_SURFACE_EXPONENTS[surface_type]
    elif surface_type is None and exponent is not None:
        # use the provided exponent
        pass
    else:
        raise ValueError(
            "Either a 'surface_type' or an 'exponent' parameter must be given")

    n, (ws, hd, hr) = vectors(wind_speed_reference, height_desired,
                               height_reference)
    adjusted = out(n)
    lib().mpv_windspeed_powerlaw(
        addr(ws), addr(hd), addr(hr), float(exponent), addr(adjusted), n,
    )
    return shaped(adjusted, wind_speed_reference, height_reference, height_desired)
