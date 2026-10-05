"""pvlib.atmosphere: airmass and the clear-sky turbidity helpers.

Port of `pvlib/atmosphere.py` in upstream's source order. `get_relative_airmass`
selects a model from a string in upstream; the dispatch happens on the host and
the model reaches the kernel as the integer the Python module documents.
"""

from std.math import log, exp, sin, cos, tan, pi

from types import F64, nan_, parallelize, par_workers, PAR_MIN_CHEAP, PAR_MIN_MEDIUM
from tools import radians_

# model ids, in the order pvlib.atmosphere.get_relative_airmass tests them
comptime AIRMASS_KASTENYOUNG1989 = 0
comptime AIRMASS_KASTEN1966 = 1
comptime AIRMASS_SIMPLE = 2
comptime AIRMASS_PICKERING2002 = 3
comptime AIRMASS_YOUNGIRVINE1967 = 4
comptime AIRMASS_YOUNG1994 = 5
comptime AIRMASS_GUEYMARD1993 = 6
comptime AIRMASS_GUEYMARD2003 = 7


def pres2alt(pressure: F64, alt: F64, n: Int):
    '''
    Determine altitude from site pressure.

    Assumes 101325 Pa at zero altitude, 288.15 K, 9.80665 m/s^2, a -6.5E-3
    K/m lapse rate, 287.053 J/(kg K) for air and 0% relative humidity.
    '''
    def one(i: Int) {pressure, alt}:
        alt.unsafe_store(i, 44331.5 - 4946.62 * pressure.unsafe_load(i) ** (0.190263))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def alt2pres(altitude: F64, press: F64, n: Int):
    '''
    Determine site pressure from altitude.
    '''
    def one(i: Int) {altitude, press}:
        press.unsafe_store(i, 100 * ((44331.514 - altitude.unsafe_load(i)) / 11880.516) ** (1 / 0.1902632))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def absolute_airmass_(airmass_relative: Float64, pressure: Float64) -> Float64:
    """`get_absolute_airmass` for one timestamp."""
    return airmass_relative * pressure / 101325.


def get_absolute_airmass(airmass_relative: F64, pressure: F64, airmass_absolute: F64, n: Int):
    r"""
    Determine absolute (pressure-adjusted) airmass from relative
    airmass and pressure.
    """
    # Memory-bound: no transcendental to spread over the cores, so the thread
    # launch would cost more than the pass.
    for i in range(n):
        airmass_absolute.unsafe_store(i, absolute_airmass_(airmass_relative.unsafe_load(i), pressure.unsafe_load(i)))


def relative_airmass_(zenith: Float64, model: Int) -> Float64:
    """`get_relative_airmass` for one timestamp."""
    # set zenith values greater than 90 to nans
    var z = zenith
    if z > 90:
        z = nan_()
    var zenith_rad = radians_(z)

    if model == AIRMASS_KASTENYOUNG1989:
        return 1.0 / (cos(zenith_rad) +
                      0.50572 * ((6.07995 + (90 - z)) ** -1.6364))
    elif model == AIRMASS_KASTEN1966:
        return 1.0 / (cos(zenith_rad) + 0.15 * ((93.885 - z) ** -1.253))
    elif model == AIRMASS_SIMPLE:
        return 1.0 / cos(zenith_rad)
    elif model == AIRMASS_PICKERING2002:
        return 1.0 / sin(radians_(90 - z + 244.0 / (165 + 47.0 * (90 - z) ** 1.1)))
    elif model == AIRMASS_YOUNGIRVINE1967:
        var sec_zen = 1.0 / cos(zenith_rad)
        return sec_zen * (1 - 0.0012 * (sec_zen * sec_zen - 1))
    elif model == AIRMASS_YOUNG1994:
        return ((1.002432 * ((cos(zenith_rad)) ** 2) +
                 0.148386 * (cos(zenith_rad)) + 0.0096467) /
                 (cos(zenith_rad) ** 3 +
                 0.149864 * (cos(zenith_rad) ** 2) +
                 0.0102963 * (cos(zenith_rad)) + 0.000303978))
    elif model == AIRMASS_GUEYMARD1993:  # [4], Eq. 22 and [5], Eq. 3b
        return 1.0 / (cos(zenith_rad) +
                      0.00176759 * (z) * ((94.37515 - z) ** -1.21563))
    elif model == AIRMASS_GUEYMARD2003:
        return 1.0 / (cos(zenith_rad) +
                      0.48353 * (z**0.095846) / (96.741 - z) ** 1.754)
    else:
        return nan_()


def get_relative_airmass(zenith: F64, airmass: F64, n: Int, model: Int):
    '''
    Calculate relative (not pressure-adjusted) airmass at sea level.
    '''
    def one(i: Int) {zenith, airmass, model}:
        airmass.unsafe_store(i, relative_airmass_(zenith.unsafe_load(i), model))

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def gueymard94_pw(temp_air: F64, relative_humidity: F64, pw: F64, n: Int):
    r"""
    Calculates precipitable water (cm) from ambient air temperature (C)
    and relatively humidity (%) using an empirical model.
    """
    def one(i: Int) {temp_air, relative_humidity, pw}:
        var T = temp_air.unsafe_load(i) + 273.15  # Convert to Kelvin
        var RH = relative_humidity.unsafe_load(i)

        var theta = T / 273.15

        # Eq. 1 from Keogh and Blakers
        var p = (
            0.1 *
            (0.4976 + 1.5265 * theta + exp(13.6897 * theta - 14.9188 * (theta)**3)) *
            (216.7 * RH / (100 * T) * exp(22.330 - 49.140 * (100 / T) -
             10.922 * (100 / T)**2 - 0.39015 * T / 100)))

        if p < 0.1:
            p = 0.1
        pw.unsafe_store(i, p)

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def bird_hulstrom80_aod_bb(aod380: F64, aod500: F64, aod_bb: F64, n: Int):
    """
    Approximate broadband aerosol optical depth.

    Bird and Hulstrom developed a correlation for broadband aerosol optical
    depth (AOD) using two wavelengths, 380 nm and 500 nm.
    """
    # approximate broadband AOD using (Bird-Hulstrom 1980)
    # approximate broadband AOD using (Bird-Hulstrom 1980)
    # Memory-bound, like the other transcendental-free maps here.
    for i in range(n):
        aod_bb.unsafe_store(i, 0.27583 * aod380.unsafe_load(i) + 0.35 * aod500.unsafe_load(i))


def kasten96_lt(
    airmass_absolute: F64, precipitable_water: F64, aod_bb: F64, lt: F64, n: Int
):
    """
    Calculate Linke turbidity  using Kasten pyrheliometric formula.
    """
    def one(i: Int) {airmass_absolute, precipitable_water, aod_bb, lt}:
        var am = airmass_absolute.unsafe_load(i)
        # other than the water vapor) the following expression"
        # - P. Ineichen (2008)
        var delta_cda = -0.101 + 0.235 * am**(-0.16)
        # "and the broadband water vapor optical depth where pwat is the integrated
        # precipitable water vapor content of the atmosphere expressed in cm and am
        # the optical air mass. The precision of these fits is better than 1% when
        # compared with Modtran simulations in the range 1 < am < 5 and
        # 0 < pwat < 5 cm at sea level" - P. Ineichen (2008)
        var delta_w = 0.112 * am**(-0.55) * precipitable_water.unsafe_load(i)**0.34
        # broadband AOD
        var delta_a = aod_bb.unsafe_load(i)
        # "Then using the Kasten pyrheliometric formula (1980, 1996), the Linke
        # turbidity at am = 2 can be written. The extension of the Linke turbidity
        # coefficient to other values of air mass was published by Ineichen and
        # Perez (2002)" - P. Ineichen (2008)
        lt.unsafe_store(i, -(9.4 + 0.9 * am) * log(exp(-am * (delta_cda + delta_w + delta_a))) / am)

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def rh_from_tdew(
    temp_air: F64,
    temp_dew: F64,
    relative_humidity: F64,
    n: Int,
    coeff0: Float64,
    coeff1: Float64,
    coeff2: Float64,
):
    """
    Calculate relative humidity from dewpoint temperature using the Magnus
    equation.
    """
    def one(i: Int) {temp_air, temp_dew, relative_humidity, coeff0, coeff1, coeff2}:
        var td = temp_dew.unsafe_load(i)
        var ta = temp_air.unsafe_load(i)
        var e = coeff0 * exp((coeff1 * td) / (coeff2 + td))
        var es = coeff0 * exp((coeff1 * ta) / (coeff2 + ta))
        relative_humidity.unsafe_store(i, 100 * (e / es))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def tdew_from_rh(
    temp_air: F64,
    relative_humidity: F64,
    dewpoint: F64,
    n: Int,
    coeff0: Float64,
    coeff1: Float64,
    coeff2: Float64,
):
    """
    Calculate dewpoint temperature using the Magnus equation; the reversal of
    `rh_from_tdew`.
    """
    def one(i: Int) {temp_air, relative_humidity, dewpoint, coeff0, coeff1, coeff2}:
        var ta = temp_air.unsafe_load(i)
        var rh = relative_humidity.unsafe_load(i)
        var ln_term = (coeff1 * ta) / (coeff2 + ta) + log(rh / 100)
        dewpoint.unsafe_store(i, coeff2 * ln_term / (coeff1 - ln_term))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def angstrom_aod_at_lambda(
    aod0: F64, lambda0: F64, aod1: F64, alpha: F64, lambda1: F64, n: Int
):
    """Get AOD at specified wavelength using Angstrom turbidity model.

    `alpha` and `lambda1` are buffers rather than kernel scalars because numpy
    broadcasts them against the series, so a per-element `alpha` is legal input
    and reading only its first element would answer for the wrong value.
    """
    def one(i: Int) {aod0, lambda0, aod1, alpha, lambda1}:
        aod1.unsafe_store(
            i,
            aod0.unsafe_load(i)
            * ((lambda1.unsafe_load(i) / lambda0.unsafe_load(i))
               ** (-alpha.unsafe_load(i))),
        )

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def angstrom_alpha(
    aod1: F64, lambda1: F64, aod2: F64, lambda2: F64, alpha_out: F64, n: Int
):
    """
    Calculate Angstrom alpha exponent.
    """
    def one(i: Int) {aod1, lambda1, aod2, lambda2, alpha_out}:
        alpha_out.unsafe_store(
            i,
            -log(aod1.unsafe_load(i) / aod2.unsafe_load(i))
            / log(lambda1.unsafe_load(i) / lambda2.unsafe_load(i)),
        )

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def windspeed_powerlaw(
    wind_speed_reference: F64,
    height_desired: F64,
    height_reference: F64,
    exponent: Float64,
    wind_speed: F64,
    n: Int,
):
    """
    Estimate wind speed for different heights.

    The model is based on the power law equation by Hellmann. `height_desired`
    and `height_reference` are per-element buffers so the ratio broadcasts the
    way `wind_speed_reference * ((height_desired / height_reference) **
    exponent)` does under numpy.
    """
    def one(i: Int) {wind_speed_reference, height_desired, height_reference, exponent, wind_speed}:
        var ws = wind_speed_reference.unsafe_load(i) * (
            (height_desired.unsafe_load(i) / height_reference.unsafe_load(i)) ** exponent
        )
        # np.where(np.iscomplex(wind_speed) | (wind_speed < 0), np.nan, wind_speed)
        # A negative ratio under a fractional exponent is complex; there is no
        # complex dtype here, so a NaN ratio or a negative product is the NaN.
        var ratio = height_desired.unsafe_load(i) / height_reference.unsafe_load(i)
        if ratio < 0.0 or ws < 0.0:
            ws = nan_()
        wind_speed.unsafe_store(i, ws)

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())
