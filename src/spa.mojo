"""pvlib.spa: the NREL Solar Position Algorithm.

Port of `pvlib/spa.py` in upstream's source order, function for function. The
only structural change is the coefficient tables: upstream keeps them as
module-level `np.array`s, Mojo has no module-level mutable state, so they live
in a host-owned float64 buffer laid out by `spa_tables` and indexed through the
`comptime` offsets emitted alongside it.
"""

from std.math import sin, cos, tan, asin, acos, atan, atan2, exp, sqrt, pi

from types import F64, parallelize, par_workers, PAR_MIN_CHEAP, PAR_MIN_TINY
from tools import radians_, degrees_
from spa_tables import (
    L0_OFF, L0_ROWS, L1_OFF, L1_ROWS, L2_OFF, L2_ROWS, L3_OFF, L3_ROWS,
    L4_OFF, L4_ROWS, L5_OFF, L5_ROWS, B0_OFF, B0_ROWS, B1_OFF, B1_ROWS,
    R0_OFF, R0_ROWS, R1_OFF, R1_ROWS, R2_OFF, R2_ROWS, R3_OFF, R3_ROWS,
    R4_OFF, R4_ROWS, NUTATION_ABCD_ARRAY_OFF, NUTATION_ABCD_ARRAY_ROWS,
    NUTATION_YTERM_ARRAY_OFF, NUTATION_YTERM_ARRAY_ROWS,
)


def julian_day(unixtime: Float64) -> Float64:
    var jd = unixtime * 1.0 / 86400 + 2440587.5
    return jd


def julian_ephemeris_day(julian_day: Float64, delta_t: Float64) -> Float64:
    var jde = julian_day + delta_t * 1.0 / 86400
    return jde


def julian_century(julian_day: Float64) -> Float64:
    var jc = (julian_day - 2451545) * 1.0 / 36525
    return jc


def julian_ephemeris_century(julian_ephemeris_day: Float64) -> Float64:
    var jce = (julian_ephemeris_day - 2451545) * 1.0 / 36525
    return jce


def julian_ephemeris_millennium(julian_ephemeris_century: Float64) -> Float64:
    var jme = julian_ephemeris_century * 1.0 / 10
    return jme


# omit type signature here; specifying read-only arrays requires use of the
# numba.types API, meaning numba must be available to import.
def sum_mult_cos_add_mult(arr: F64, nrows: Int, x: Float64) -> Float64:
    # shared calculation used for heliocentric longitude, latitude, and radius
    var s = 0.0
    for row in range(nrows):
        s += arr.unsafe_load(row * 3) * cos(arr.unsafe_load(row * 3 + 1) + arr.unsafe_load(row * 3 + 2) * x)
    return s


def heliocentric_longitude(tb: F64, jme: Float64) -> Float64:
    var l0 = sum_mult_cos_add_mult(tb.unsafe_offset(L0_OFF), L0_ROWS, jme)
    var l1 = sum_mult_cos_add_mult(tb.unsafe_offset(L1_OFF), L1_ROWS, jme)
    var l2 = sum_mult_cos_add_mult(tb.unsafe_offset(L2_OFF), L2_ROWS, jme)
    var l3 = sum_mult_cos_add_mult(tb.unsafe_offset(L3_OFF), L3_ROWS, jme)
    var l4 = sum_mult_cos_add_mult(tb.unsafe_offset(L4_OFF), L4_ROWS, jme)
    var l5 = sum_mult_cos_add_mult(tb.unsafe_offset(L5_OFF), L5_ROWS, jme)

    var l_rad = (l0 + l1 * jme + l2 * jme**2 + l3 * jme**3 + l4 * jme**4 +
                 l5 * jme**5) / 10**8
    var l = degrees_(l_rad)
    return l % 360


def heliocentric_latitude(tb: F64, jme: Float64) -> Float64:
    var b0 = sum_mult_cos_add_mult(tb.unsafe_offset(B0_OFF), B0_ROWS, jme)
    var b1 = sum_mult_cos_add_mult(tb.unsafe_offset(B1_OFF), B1_ROWS, jme)

    var b_rad = (b0 + b1 * jme) / 10**8
    var b = degrees_(b_rad)
    return b


def heliocentric_radius_vector(tb: F64, jme: Float64) -> Float64:
    var r0 = sum_mult_cos_add_mult(tb.unsafe_offset(R0_OFF), R0_ROWS, jme)
    var r1 = sum_mult_cos_add_mult(tb.unsafe_offset(R1_OFF), R1_ROWS, jme)
    var r2 = sum_mult_cos_add_mult(tb.unsafe_offset(R2_OFF), R2_ROWS, jme)
    var r3 = sum_mult_cos_add_mult(tb.unsafe_offset(R3_OFF), R3_ROWS, jme)
    var r4 = sum_mult_cos_add_mult(tb.unsafe_offset(R4_OFF), R4_ROWS, jme)

    var r = (r0 + r1 * jme + r2 * jme**2 + r3 * jme**3 + r4 * jme**4) / 10**8
    return r


def geocentric_longitude(heliocentric_longitude: Float64) -> Float64:
    var theta = heliocentric_longitude + 180.0
    return theta % 360


def geocentric_latitude(heliocentric_latitude: Float64) -> Float64:
    var beta = -1.0 * heliocentric_latitude
    return beta


def mean_elongation(julian_ephemeris_century: Float64) -> Float64:
    var x0 = (297.85036
              + 445267.111480 * julian_ephemeris_century
              - 0.0019142 * julian_ephemeris_century**2
              + julian_ephemeris_century**3 / 189474)
    return x0


def mean_anomaly_sun(julian_ephemeris_century: Float64) -> Float64:
    var x1 = (357.52772
              + 35999.050340 * julian_ephemeris_century
              - 0.0001603 * julian_ephemeris_century**2
              - julian_ephemeris_century**3 / 300000)
    return x1


def mean_anomaly_moon(julian_ephemeris_century: Float64) -> Float64:
    var x2 = (134.96298
              + 477198.867398 * julian_ephemeris_century
              + 0.0086972 * julian_ephemeris_century**2
              + julian_ephemeris_century**3 / 56250)
    return x2


def moon_argument_latitude(julian_ephemeris_century: Float64) -> Float64:
    var x3 = (93.27191
              + 483202.017538 * julian_ephemeris_century
              - 0.0036825 * julian_ephemeris_century**2
              + julian_ephemeris_century**3 / 327270)
    return x3


def moon_ascending_longitude(julian_ephemeris_century: Float64) -> Float64:
    var x4 = (125.04452
              - 1934.136261 * julian_ephemeris_century
              + 0.0020708 * julian_ephemeris_century**2
              + julian_ephemeris_century**3 / 450000)
    return x4


def longitude_obliquity_nutation(
    julian_ephemeris_century: Float64, x0: Float64, x1: Float64, x2: Float64,
    x3: Float64, x4: Float64, tb: F64
) -> Tuple[Float64, Float64]:
    var delta_psi_sum = 0.0
    var delta_eps_sum = 0.0
    var abcd = tb.unsafe_offset(NUTATION_ABCD_ARRAY_OFF)
    var yterm = tb.unsafe_offset(NUTATION_YTERM_ARRAY_OFF)
    for row in range(NUTATION_YTERM_ARRAY_ROWS):
        var a = abcd.unsafe_load(row * 4)
        var b = abcd.unsafe_load(row * 4 + 1)
        var c = abcd.unsafe_load(row * 4 + 2)
        var d = abcd.unsafe_load(row * 4 + 3)
        var arg = radians_(
            yterm.unsafe_load(row * 5) * x0 +
            yterm.unsafe_load(row * 5 + 1) * x1 +
            yterm.unsafe_load(row * 5 + 2) * x2 +
            yterm.unsafe_load(row * 5 + 3) * x3 +
            yterm.unsafe_load(row * 5 + 4) * x4
        )
        delta_psi_sum += (a + b * julian_ephemeris_century) * sin(arg)
        delta_eps_sum += (c + d * julian_ephemeris_century) * cos(arg)
    var delta_psi = delta_psi_sum * 1.0 / 36000000
    var delta_eps = delta_eps_sum * 1.0 / 36000000
    return (delta_psi, delta_eps)

def mean_ecliptic_obliquity(julian_ephemeris_millennium: Float64) -> Float64:
    var U = 1.0 * julian_ephemeris_millennium / 10
    var e0 = (84381.448 - 4680.93 * U - 1.55 * U**2
              + 1999.25 * U**3 - 51.38 * U**4 - 249.67 * U**5
              - 39.05 * U**6 + 7.12 * U**7 + 27.87 * U**8
              + 5.79 * U**9 + 2.45 * U**10)
    return e0


def true_ecliptic_obliquity(
    mean_ecliptic_obliquity: Float64, obliquity_nutation: Float64
) -> Float64:
    var e0 = mean_ecliptic_obliquity
    var deleps = obliquity_nutation
    var e = e0 * 1.0 / 3600 + deleps
    return e


def aberration_correction(earth_radius_vector: Float64) -> Float64:
    var deltau = -20.4898 / (3600 * earth_radius_vector)
    return deltau


def apparent_sun_longitude(
    geocentric_longitude: Float64, longitude_nutation: Float64,
    aberration_correction: Float64
) -> Float64:
    var lamd = geocentric_longitude + longitude_nutation + aberration_correction
    return lamd


def mean_sidereal_time(julian_day: Float64, julian_century: Float64) -> Float64:
    var v0 = (280.46061837 + 360.98564736629 * (julian_day - 2451545)
              + 0.000387933 * julian_century**2 - julian_century**3 / 38710000)
    return v0 % 360.0


def apparent_sidereal_time(
    mean_sidereal_time: Float64, longitude_nutation: Float64,
    true_ecliptic_obliquity: Float64
) -> Float64:
    var v = mean_sidereal_time + longitude_nutation * cos(
        radians_(true_ecliptic_obliquity)
    )
    return v


def geocentric_sun_right_ascension(
    apparent_sun_longitude: Float64, true_ecliptic_obliquity: Float64,
    geocentric_latitude: Float64
) -> Float64:
    var true_ecliptic_obliquity_rad = radians_(true_ecliptic_obliquity)
    var apparent_sun_longitude_rad = radians_(apparent_sun_longitude)

    var num = (sin(apparent_sun_longitude_rad)
               * cos(true_ecliptic_obliquity_rad)
               - tan(radians_(geocentric_latitude))
               * sin(true_ecliptic_obliquity_rad))
    var alpha = degrees_(atan2(num, cos(apparent_sun_longitude_rad)))
    return alpha % 360


def geocentric_sun_declination(
    apparent_sun_longitude: Float64, true_ecliptic_obliquity: Float64,
    geocentric_latitude: Float64
) -> Float64:
    var geocentric_latitude_rad = radians_(geocentric_latitude)
    var true_ecliptic_obliquity_rad = radians_(true_ecliptic_obliquity)

    var delta = degrees_(asin(
        sin(geocentric_latitude_rad) * cos(true_ecliptic_obliquity_rad) +
        cos(geocentric_latitude_rad) * sin(true_ecliptic_obliquity_rad) *
        sin(radians_(apparent_sun_longitude))
    ))
    return delta


def local_hour_angle(
    apparent_sidereal_time: Float64, observer_longitude: Float64,
    sun_right_ascension: Float64
) -> Float64:
    """Measured westward from south"""
    var H = apparent_sidereal_time + observer_longitude - sun_right_ascension
    return H % 360


def equatorial_horizontal_parallax(earth_radius_vector: Float64) -> Float64:
    var xi = 8.794 / (3600 * earth_radius_vector)
    return xi


def uterm(observer_latitude: Float64) -> Float64:
    var u = atan(0.99664719 * tan(radians_(observer_latitude)))
    return u


def xterm(u: Float64, observer_latitude: Float64, observer_elevation: Float64) -> Float64:
    var x = (cos(u) + observer_elevation / 6378140
             * cos(radians_(observer_latitude)))
    return x


def yterm(u: Float64, observer_latitude: Float64, observer_elevation: Float64) -> Float64:
    var y = (0.99664719 * sin(u) + observer_elevation / 6378140
             * sin(radians_(observer_latitude)))
    return y


def parallax_sun_right_ascension(
    xterm: Float64, equatorial_horizontal_parallax: Float64,
    local_hour_angle: Float64, geocentric_sun_declination: Float64
) -> Float64:
    var equatorial_horizontal_parallax_rad = radians_(equatorial_horizontal_parallax)
    var local_hour_angle_rad = radians_(local_hour_angle)

    var num = (-xterm * sin(equatorial_horizontal_parallax_rad)
               * sin(local_hour_angle_rad))
    var denom = (cos(radians_(geocentric_sun_declination))
                 - xterm * sin(equatorial_horizontal_parallax_rad)
                 * cos(local_hour_angle_rad))
    var delta_alpha = degrees_(atan2(num, denom))
    return delta_alpha


def topocentric_sun_right_ascension(
    geocentric_sun_right_ascension: Float64, parallax_sun_right_ascension: Float64
) -> Float64:
    var alpha_prime = geocentric_sun_right_ascension + parallax_sun_right_ascension
    return alpha_prime


def topocentric_sun_declination(
    geocentric_sun_declination: Float64, xterm: Float64, yterm: Float64,
    equatorial_horizontal_parallax: Float64, parallax_sun_right_ascension: Float64,
    local_hour_angle: Float64
) -> Float64:
    var geocentric_sun_declination_rad = radians_(geocentric_sun_declination)
    var equatorial_horizontal_parallax_rad = radians_(equatorial_horizontal_parallax)

    var num = ((sin(geocentric_sun_declination_rad) - yterm
               * sin(equatorial_horizontal_parallax_rad))
               * cos(radians_(parallax_sun_right_ascension)))
    var denom = (cos(geocentric_sun_declination_rad) - xterm
                 * sin(equatorial_horizontal_parallax_rad)
                 * cos(radians_(local_hour_angle)))
    var delta = degrees_(atan2(num, denom))
    return delta


def topocentric_local_hour_angle(
    local_hour_angle: Float64, parallax_sun_right_ascension: Float64
) -> Float64:
    var H_prime = local_hour_angle - parallax_sun_right_ascension
    return H_prime


def topocentric_elevation_angle_without_atmosphere(
    observer_latitude: Float64, topocentric_sun_declination: Float64,
    topocentric_local_hour_angle: Float64
) -> Float64:
    var observer_latitude_rad = radians_(observer_latitude)
    var topocentric_sun_declination_rad = radians_(topocentric_sun_declination)
    var e0 = degrees_(asin(
        sin(observer_latitude_rad) * sin(topocentric_sun_declination_rad) +
        cos(observer_latitude_rad) * cos(topocentric_sun_declination_rad)
        * cos(radians_(topocentric_local_hour_angle))
    ))
    return e0


def atmospheric_refraction_correction(
    local_pressure: Float64, local_temp: Float64,
    topocentric_elevation_angle_wo_atmosphere: Float64, atmos_refract: Float64
) -> Float64:
    # switch sets delta_e when the sun is below the horizon
    var switch = topocentric_elevation_angle_wo_atmosphere >= -1.0 * (
        0.26667 + atmos_refract
    )
    var delta_e = ((local_pressure / 1010.0) * (283.0 / (273 + local_temp))
                   * 1.02 / (60 * tan(radians_(
                       topocentric_elevation_angle_wo_atmosphere
                       + 10.3 / (topocentric_elevation_angle_wo_atmosphere + 5.11)
                   )))) * Float64(switch)
    return delta_e


def topocentric_elevation_angle(
    topocentric_elevation_angle_without_atmosphere: Float64,
    atmospheric_refraction_correction: Float64
) -> Float64:
    var e = (topocentric_elevation_angle_without_atmosphere
             + atmospheric_refraction_correction)
    return e


def topocentric_zenith_angle(topocentric_elevation_angle: Float64) -> Float64:
    var theta = 90 - topocentric_elevation_angle
    return theta


def topocentric_astronomers_azimuth(
    topocentric_local_hour_angle: Float64, topocentric_sun_declination: Float64,
    observer_latitude: Float64
) -> Float64:
    var topocentric_local_hour_angle_rad = radians_(topocentric_local_hour_angle)
    var observer_latitude_rad = radians_(observer_latitude)
    var num = sin(topocentric_local_hour_angle_rad)
    var denom = (cos(topocentric_local_hour_angle_rad)
                 * sin(observer_latitude_rad)
                 - tan(radians_(topocentric_sun_declination))
                 * cos(observer_latitude_rad))
    var gamma = degrees_(atan2(num, denom))
    return gamma % 360


def topocentric_azimuth_angle(topocentric_astronomers_azimuth: Float64) -> Float64:
    var phi = topocentric_astronomers_azimuth + 180
    return phi % 360


def sun_mean_longitude(julian_ephemeris_millennium: Float64) -> Float64:
    var M = (280.4664567 + 360007.6982779 * julian_ephemeris_millennium
             + 0.03032028 * julian_ephemeris_millennium**2
             + julian_ephemeris_millennium**3 / 49931
             - julian_ephemeris_millennium**4 / 15300
             - julian_ephemeris_millennium**5 / 2000000)
    return M


def equation_of_time(
    sun_mean_longitude: Float64, geocentric_sun_right_ascension: Float64,
    longitude_nutation: Float64, true_ecliptic_obliquity: Float64
) -> Float64:
    var E = (sun_mean_longitude - 0.0057183 - geocentric_sun_right_ascension +
             longitude_nutation * cos(radians_(true_ecliptic_obliquity)))
    # limit between 0 and 360
    E = E % 360
    # convert to minutes
    E *= 4
    var greater = E > 20
    var less = E < -20
    var other = (E <= 20) and (E >= -20)
    E = Float64(greater) * (E - 1440) + Float64(less) * (E + 1440) + Float64(other) * E
    return E


def solar_position_loop(
    unixtime: F64, delta_t: F64, n: Int,
    lat: Float64, lon: Float64, elev: Float64, pressure: Float64,
    temp: Float64, atmos_refract: Float64, sst: Int, esd: Int,
    theta: F64, theta0: F64, e: F64, e0: F64, phi: F64, eot: F64,
    tb: F64,
):
    """Loop through the time array and calculate the solar position"""
    def one(i: Int) {unixtime, delta_t, lat, lon, elev, pressure, temp,
                      atmos_refract, sst, esd, theta, theta0, e, e0, phi, eot,
                      tb}:
        var utime = unixtime.unsafe_load(i)
        var dT = delta_t.unsafe_load(i)
        var jd = julian_day(utime)
        var jde = julian_ephemeris_day(jd, dT)
        var jc = julian_century(jd)
        var jce = julian_ephemeris_century(jde)
        var jme = julian_ephemeris_millennium(jce)
        var R = heliocentric_radius_vector(tb, jme)
        if esd != 0:
            theta.unsafe_store(i, R)
            return
        var L = heliocentric_longitude(tb, jme)
        var B = heliocentric_latitude(tb, jme)
        var Theta = geocentric_longitude(L)
        var beta = geocentric_latitude(B)
        var x0 = mean_elongation(jce)
        var x1 = mean_anomaly_sun(jce)
        var x2 = mean_anomaly_moon(jce)
        var x3 = moon_argument_latitude(jce)
        var x4 = moon_ascending_longitude(jce)
        var nutation = longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4, tb)
        var delta_psi = nutation[0]
        var delta_epsilon = nutation[1]
        var epsilon0 = mean_ecliptic_obliquity(jme)
        var epsilon = true_ecliptic_obliquity(epsilon0, delta_epsilon)
        var delta_tau = aberration_correction(R)
        var lamd = apparent_sun_longitude(Theta, delta_psi, delta_tau)
        var v0 = mean_sidereal_time(jd, jc)
        var v = apparent_sidereal_time(v0, delta_psi, epsilon)
        var alpha = geocentric_sun_right_ascension(lamd, epsilon, beta)
        var delta = geocentric_sun_declination(lamd, epsilon, beta)
        if sst != 0:
            theta.unsafe_store(i, v)
            theta0.unsafe_store(i, alpha)
            e.unsafe_store(i, delta)
            return
        var m = sun_mean_longitude(jme)
        var eot_ = equation_of_time(m, alpha, delta_psi, epsilon)
        var H = local_hour_angle(v, lon, alpha)
        var xi = equatorial_horizontal_parallax(R)
        var u = uterm(lat)
        var x = xterm(u, lat, elev)
        var y = yterm(u, lat, elev)
        var delta_alpha = parallax_sun_right_ascension(x, xi, H, delta)
        var delta_prime = topocentric_sun_declination(delta, x, y, xi, delta_alpha, H)
        var H_prime = topocentric_local_hour_angle(H, delta_alpha)
        var e0_ = topocentric_elevation_angle_without_atmosphere(lat, delta_prime, H_prime)
        var delta_e = atmospheric_refraction_correction(pressure, temp, e0_, atmos_refract)
        var e_ = topocentric_elevation_angle(e0_, delta_e)
        var theta_ = topocentric_zenith_angle(e_)
        var theta0_ = topocentric_zenith_angle(e0_)
        var gamma = topocentric_astronomers_azimuth(H_prime, delta_prime, lat)
        var phi_ = topocentric_azimuth_angle(gamma)
        theta.unsafe_store(i, theta_)
        theta0.unsafe_store(i, theta0_)
        e.unsafe_store(i, e_)
        e0.unsafe_store(i, e0_)
        phi.unsafe_store(i, phi_)
        eot.unsafe_store(i, eot_)

    if n < PAR_MIN_TINY:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def calculate_deltat(year: F64, month: F64, deltat: F64, n: Int):
    """Calculate the difference between Terrestrial Dynamical Time (TD)
    and Universal Time (UT).

    Equations taken from http://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html
    """
    def one(k: Int) {year, month, deltat}:
        var yr = year.unsafe_load(k)
        var y = yr + (month.unsafe_load(k) - 0.5) / 12
        var d = 0.0
        var c = y / 100

        if yr < -500:
            d = -20 + 32 * ((y - 1820) / 100) ** 2

        elif (yr >= -500) and (yr < 500):
            d = (10583.6 - 1014.41 * c
                 + 33.78311 * c**2
                 - 5.952053 * c**3
                 - 0.1798452 * c**4
                 + 0.022174192 * c**5
                 + 0.0090316521 * c**6)

        elif (yr >= 500) and (yr < 1600):
            var u = (y - 1000) / 100
            d = (1574.2 - 556.01 * u
                 + 71.23472 * u**2
                 + 0.319781 * u**3
                 - 0.8503463 * u**4
                 - 0.005050998 * u**5
                 + 0.0083572073 * u**6)

        elif (yr >= 1600) and (yr < 1700):
            var u = y - 1600
            d = (120 - 0.9808 * u - 0.01532 * u**2 + u**3 / 7129)

        elif (yr >= 1700) and (yr < 1800):
            var u = y - 1700
            d = (8.83 + 0.1603 * u
                 - 0.0059285 * u**2
                 + 0.00013336 * u**3
                 - u**4 / 1174000)

        elif (yr >= 1800) and (yr < 1860):
            var u = y - 1800
            d = (13.72 - 0.332447 * u
                 + 0.0068612 * u**2
                 + 0.0041116 * u**3
                 - 0.00037436 * u**4
                 + 0.0000121272 * u**5
                 - 0.0000001699 * u**6
                 + 0.000000000875 * u**7)

        elif (yr >= 1860) and (yr < 1900):
            var u = y - 1860
            d = (7.62 + 0.5737 * u
                 - 0.251754 * u**2
                 + 0.01680668 * u**3
                 - 0.0004473624 * u**4
                 + u**5 / 233174)

        elif (yr >= 1900) and (yr < 1920):
            var u = y - 1900
            d = (-2.79 + 1.494119 * u
                 - 0.0598939 * u**2
                 + 0.0061966 * u**3
                 - 0.000197 * u**4)

        elif (yr >= 1920) and (yr < 1941):
            var u = y - 1920
            d = (21.20 + 0.84493 * u - 0.076100 * u**2 + 0.0020936 * u**3)

        elif (yr >= 1941) and (yr < 1961):
            var u = y - 1950
            d = (29.07 + 0.407 * u - u**2 / 233 + u**3 / 2547)

        elif (yr >= 1961) and (yr < 1986):
            var u = y - 1975
            d = (45.45 + 1.067 * u - u**2 / 260 - u**3 / 718)

        elif (yr >= 1986) and (yr < 2005):
            var u = y - 2000
            d = (63.86 + 0.3345 * u
                 - 0.060374 * u**2
                 + 0.0017275 * u**3
                 + 0.000651814 * u**4
                 + 0.00002373599 * u**5)

        elif (yr >= 2005) and (yr < 2050):
            var u = y - 2000
            d = 62.92 + 0.32217 * u + 0.005589 * u**2

        elif (yr >= 2050) and (yr < 2150):
            d = (-20 + 32 * ((y - 1820) / 100) ** 2) - 0.5628 * (2150 - y)

        elif yr >= 2150:
            d = -20 + 32 * ((y - 1820) / 100) ** 2

        deltat.unsafe_store(k, d)

    if n < PAR_MIN_CHEAP:
        for k in range(n):
            one(k)
        return
    parallelize(one, n, par_workers())
