"""pvlib.solarposition: the analytical solar position models.

Port of the array-valued functions in `pvlib/solarposition.py`, in upstream's
source order. Upstream leans on numpy broadcasting; here each of these is the
same straight-line elementwise map over flat float64 buffers, and the host
broadcasts a scalar argument to the length of the others.
"""

from std.math import sin, cos, tan, acos, pi, isnan

from types import F64, parallelize, par_workers, PAR_MIN_CHEAP, PAR_MIN_MEDIUM
from tools import radians_, degrees_


def _day_angle(dayofyear: Float64, offset: Float64) -> Float64:
	    """`dayofyear` shifted by Spencer's 1 (ASCE uses 0), scaled to radians."""
	    return (2. * pi / 365.) * (dayofyear - offset)


def _calculate_simple_day_angle(dayofyear: F64, offset: Float64, day_angle: F64, n: Int):
	    """
	    Calculates the day angle for the Earth's orbit around the Sun.

	    offset: 1 for the Spencer method, 0 for the ASCE method
	    """
	    def one(i: Int) {dayofyear, day_angle, offset}:
	        day_angle.unsafe_store(i, _day_angle(dayofyear.unsafe_load(i), offset))

	    if n < PAR_MIN_MEDIUM:
	        for i in range(n):
	            one(i)
	        return
	    parallelize(one, n, par_workers())


def equation_of_time_spencer71(dayofyear: F64, eot: F64, n: Int, day_angle: F64):
	    """
	    Equation of time from Duffie & Beckman and attributed to Spencer
	    (1971) and Iqbal (1983).
	    """
	    # The day angle and the model are one pass and one launch: each costs
	    # a fixed ~5 ms thread launch, so paying it twice dominated the kernel.
	    def one(i: Int) {dayofyear, day_angle, eot}:
	        var b = _day_angle(dayofyear.unsafe_load(i), 1.0)
	        day_angle.unsafe_store(i, b)
	        # convert from radians to minutes per day = 24[h/day] * 60[min/h] / 2 / pi
	        eot.unsafe_store(i, (1440.0 / 2 / pi) * (
	            0.0000075 +
	            0.001868 * cos(b) - 0.032077 * sin(b) -
	            0.014615 * cos(2.0 * b) - 0.040849 * sin(2.0 * b)
	        ))

	    if n < PAR_MIN_MEDIUM:
	        for i in range(n):
	            one(i)
	        return
	    parallelize(one, n, par_workers())


def equation_of_time_pvcdrom(dayofyear: F64, eot: F64, n: Int, day_angle: F64):
	    """
	    Equation of time from PVCDROM.
	    """
	    # day angle relative to Vernal Equinox, typically March 22 (day number 81)
	    def one(i: Int) {dayofyear, day_angle, eot}:
	        var b = _day_angle(dayofyear.unsafe_load(i), 1.0)
	        day_angle.unsafe_store(i, b)
	        var bday = b - (2.0 * pi / 365.0) * 80.0
	        # same value but about 2x faster than Spencer (1971)
	        eot.unsafe_store(i, 9.87 * sin(2.0 * bday) - 7.53 * cos(bday) - 1.5 * sin(bday))

	    if n < PAR_MIN_MEDIUM:
	        for i in range(n):
	            one(i)
	        return
	    parallelize(one, n, par_workers())


def declination_spencer71(dayofyear: F64, dec: F64, n: Int, day_angle: F64):
	    """
	    Solar declination from Duffie & Beckman and attributed to
	    Spencer (1971) and Iqbal (1983).
	    """
	    def one(i: Int) {dayofyear, day_angle, dec}:
	        var b = _day_angle(dayofyear.unsafe_load(i), 1.0)
	        day_angle.unsafe_store(i, b)
	        dec.unsafe_store(i, (
	            0.006918 -
	            0.399912 * cos(b) + 0.070257 * sin(b) -
	            0.006758 * cos(2. * b) + 0.000907 * sin(2. * b) -
	            0.002697 * cos(3. * b) + 0.00148 * sin(3. * b)
	        ))

	    if n < PAR_MIN_MEDIUM:
	        for i in range(n):
	            one(i)
	        return
	    parallelize(one, n, par_workers())


def declination_cooper69(dayofyear: F64, dec: F64, n: Int, day_angle: F64):
	    """
	    Solar declination from Duffie & Beckman and attributed to Cooper (1969).
	    """
	    def one(i: Int) {dayofyear, day_angle, dec}:
	        var b = _day_angle(dayofyear.unsafe_load(i), 1.0)
	        day_angle.unsafe_store(i, b)
	        dec.unsafe_store(i, radians_(23.45 * sin(b + (2.0 * pi / 365.0) * 285.0)))

	    if n < PAR_MIN_MEDIUM:
	        for i in range(n):
	            one(i)
	        return
	    parallelize(one, n, par_workers())


def solar_azimuth_analytical(
    latitude: F64, hourangle: F64, declination: F64, zenith: F64,
    azimuth: F64, n: Int
):
    """
    Analytical expression of solar azimuth angle based on spherical
    trigonometry.
    """
    def one(i: Int) {latitude, hourangle, declination, zenith, azimuth}:
        # all four inputs are radians here, as upstream documents
        var lat = latitude.unsafe_load(i)
        var dec = declination.unsafe_load(i)
        var z = zenith.unsafe_load(i)
        var numer = (cos(z) * sin(lat) - sin(dec))
        var denom = (sin(z) * cos(lat))

        # cases that would generate new NaN values are safely ignored here
        # since they are dealt with further below
        var cos_azi = numer / denom

        # when zero division occurs, use the limit value of the analytical
        # expression
        if _isclose(denom, 0.0):
            cos_azi = 1.0

        # when too many round-ups in floating point math take cos_azi beyond
        # 1.0, use 1.0
        if _isclose(cos_azi, 1.0):
            cos_azi = 1.0
        if _isclose(cos_azi, -1.0):
            cos_azi = -1.0

        # when NaN values occur in input, ignore and pass to output
        var sign_ha = _sign(hourangle.unsafe_load(i))

        azimuth.unsafe_store(i, sign_ha * acos(cos_azi) + pi)

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def solar_zenith_analytical(
    latitude: F64, hourangle: F64, declination: F64, zenith: F64, n: Int
):
    """
    Analytical expression of solar zenith angle based on spherical
    trigonometry.
    """
    def one(i: Int) {latitude, hourangle, declination, zenith}:
        zenith.unsafe_store(i, acos(
            cos(declination.unsafe_load(i)) * cos(latitude.unsafe_load(i))
            * cos(hourangle.unsafe_load(i)) +
            sin(declination.unsafe_load(i)) * sin(latitude.unsafe_load(i))
        ))

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def hour_angle(
    hrs_minus_tzs: F64, longitude: Float64, equation_of_time: F64,
    hourangle: F64, n: Int
):
    """
    Hour angle in local solar time. Zero at local solar noon.
    """
    # Memory-bound: no transcendental, so the serial pass already saturates
    # the link and the thread launch would cost more than it.
    for i in range(n):
        hourangle.unsafe_store(i, 15. * (hrs_minus_tzs.unsafe_load(i) - 12.) + longitude
                        + equation_of_time.unsafe_load(i) / 4.)


def sunset_angle(latitude: Float64, declination: F64, angle: F64, n: Int):
    """
    The sunset hour angle of `sun_rise_set_transit_geometric`, in degrees.

    sunrise_hour uses the same value negated, and transit_hour uses zero.
    """
    var latitude_rad = radians_(latitude)

    def one(i: Int) {declination, angle, latitude_rad}:
        var sunset_angle_rad = acos(-tan(declination.unsafe_load(i)) * tan(latitude_rad))
        angle.unsafe_store(i, degrees_(sunset_angle_rad))  # degrees

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


# np.isclose(a, b, rtol=0.0, atol=1e-8)
def _isclose(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= 1e-8


def _sign(x: Float64) -> Float64:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return x * 0.0
