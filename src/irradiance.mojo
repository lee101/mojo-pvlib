"""pvlib.irradiance: extraterrestrial, transposition and decomposition models.

Port of `pvlib/irradiance.py` in upstream's source order. Functions that
upstream writes as a chain of whole-array numpy expressions become either one
straight-line elementwise loop or a short sequence of loops over the same
intermediate arrays, in the same order and with the same branches.
"""

from std.math import sin, cos, tan, acos, exp, sqrt, pi, isnan, isfinite

from types import (
    F64, maximum, minimum, clip, nan_, pow_, parallelize, par_workers,
    PAR_MIN_CHEAP, PAR_MIN_MEDIUM, PAR_MIN_HEAVY,
)
from tools import radians_, degrees_
from atmosphere import (
    relative_airmass_, absolute_airmass_, AIRMASS_KASTEN1966, AIRMASS_SIMPLE,
)

# get_extra_radiation methods, in the order pvlib.irradiance tests them
comptime EXTRA_RADIATION_ASCE = 0
comptime EXTRA_RADIATION_SPENCER = 1


# the scalar companions of the tools.mojo elementwise kernels
def cosd_(x: Float64) -> Float64:
    return cos(radians_(x))


def sind_(x: Float64) -> Float64:
    return sin(radians_(x))


def _get_extra_radiation(doy: Float64, solar_constant: Float64, method: Int) -> Float64:
    # consider putting asce and spencer methods in their own functions
    var RoverR0sqrd: Float64
    if method == EXTRA_RADIATION_ASCE:
        var B = (2. * pi / 365.) * (doy - 0.0)
        RoverR0sqrd = 1 + 0.033 * cos(B)
    elif method == EXTRA_RADIATION_SPENCER:
        var B = (2. * pi / 365.) * (doy - 1.0)
        RoverR0sqrd = (1.00011 + 0.034221 * cos(B) + 0.00128 * sin(B) +
                      0.000719 * cos(2 * B) + 7.7e-05 * sin(2 * B))
    else:
        RoverR0sqrd = nan_()

    return solar_constant * RoverR0sqrd


def get_extra_radiation(
    doy: F64, dni_extra: F64, n: Int, solar_constant: Float64, method: Int
):
    """
    Determine extraterrestrial radiation from day of year.
    """
    def one(i: Int) {doy, dni_extra, solar_constant, method}:
        dni_extra.unsafe_store(i, _get_extra_radiation(doy.unsafe_load(i), solar_constant, method))

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def _aoi_projection(
    surface_tilt: Float64, surface_azimuth: Float64, solar_zenith: Float64,
    solar_azimuth: Float64
) -> Float64:
    """`aoi_projection` for one timestamp."""
    var projection = (
        cosd_(surface_tilt) * cosd_(solar_zenith) +
        sind_(surface_tilt) * sind_(solar_zenith) *
        cosd_(solar_azimuth - surface_azimuth)
    )

    # GH 1185
    return clip(projection, -1, 1)



struct AoiParts:
    var projection: Float64
    var cos_tilt: Float64
    var cos_zen: Float64

    def __init__(out self, projection: Float64, cos_tilt: Float64, cos_zen: Float64):
        self.projection = projection
        self.cos_tilt = cos_tilt
        self.cos_zen = cos_zen


def _aoi_parts(
    surface_tilt: Float64, surface_azimuth: Float64, solar_zenith: Float64,
    solar_azimuth: Float64
) -> AoiParts:
    """`_aoi_projection` keeping the two cosines it already had to evaluate.

    Every caller of the transposition models needs `cosd(surface_tilt)` or
    `cosd(solar_zenith)` again right after the projection, so evaluating them
    once here removes one or two libm calls per element.
    """
    var cos_tilt = cosd_(surface_tilt)
    var cos_zen = cosd_(solar_zenith)
    var sin_tilt = sind_(surface_tilt)
    var sin_zen = sind_(solar_zenith)
    var projection = (
        cos_tilt * cos_zen +
        sin_tilt * sin_zen * cosd_(solar_azimuth - surface_azimuth)
    )
    return AoiParts(clip(projection, -1, 1), cos_tilt, cos_zen)


def aoi_projection(
    surface_tilt: F64, surface_azimuth: F64, solar_zenith: F64,
    solar_azimuth: F64, projection: F64, n: Int
):
    """
    Calculates the dot product of the sun position unit vector and the surface
    normal unit vector; in other words, the cosine of the angle of incidence.

    Usage note: When the sun is behind the surface the value returned is
    negative.  For many uses negative values must be set to zero.
    """
    def one(i: Int) {surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, projection}:
        projection.unsafe_store(i, _aoi_projection(
            surface_tilt.unsafe_load(i), surface_azimuth.unsafe_load(i),
            solar_zenith.unsafe_load(i), solar_azimuth.unsafe_load(i)
        ))

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def aoi(
    surface_tilt: F64, surface_azimuth: F64, solar_zenith: F64,
    solar_azimuth: F64, aoi: F64, n: Int
):
    """
    Calculates the angle of incidence of the solar vector on a surface.
    This is the angle between the solar vector and the surface normal.
    """
    def one(i: Int) {surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, aoi}:
        var projection = _aoi_projection(
            surface_tilt.unsafe_load(i), surface_azimuth.unsafe_load(i), solar_zenith.unsafe_load(i), solar_azimuth.unsafe_load(i)
        )
        aoi.unsafe_store(i, degrees_(acos(projection)))

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def beam_component(
    surface_tilt: F64, surface_azimuth: F64, solar_zenith: F64,
    solar_azimuth: F64, dni: F64, beam: F64, n: Int
):
    """
    Calculates the beam component of the plane of array irradiance.
    """
    def one(i: Int) {surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, beam}:
        var b = dni.unsafe_load(i) * _aoi_projection(
            surface_tilt.unsafe_load(i), surface_azimuth.unsafe_load(i), solar_zenith.unsafe_load(i), solar_azimuth.unsafe_load(i)
        )
        beam.unsafe_store(i, maximum(b, 0))

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def get_ground_diffuse(
    surface_tilt: F64, ghi: F64, poa_ground_diffuse: F64, n: Int, albedo: Float64
):
    r'''
    Estimate diffuse irradiance on a tilted surface from ground reflections.

    Ground diffuse irradiance is calculated as

    .. math::

       G_{ground} = GHI \times \rho \times \frac{1 - \cos\beta}{2}
    '''
    def one(i: Int) {surface_tilt, ghi, poa_ground_diffuse, albedo}:
        poa_ground_diffuse.unsafe_store(i, (
            ghi.unsafe_load(i) * albedo * (1 - cosd_(surface_tilt.unsafe_load(i))) * 0.5
        ))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def isotropic(
    surface_tilt: F64, dhi: F64, poa_sky_diffuse: F64, poa_isotropic: F64, n: Int
):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using
    the isotropic sky model.

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2}
    '''
    def one(i: Int) {surface_tilt, dhi, poa_sky_diffuse, poa_isotropic}:
        var sky = dhi.unsafe_load(i) * (1 + cosd_(surface_tilt.unsafe_load(i))) * 0.5
        poa_sky_diffuse.unsafe_store(i, sky)
        poa_isotropic.unsafe_store(i, sky)

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def klucher(
    surface_tilt: F64, surface_azimuth: F64, dhi: F64, ghi: F64, solar_zenith: F64,
    solar_azimuth: F64, poa_sky_diffuse: F64, n: Int
):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface
    using the Klucher (1979) model.

    The expression for the sky diffuse irradiance, :math:`I_d`, is

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2} (1 + F' \sin^3(\beta/2))
       (1 + F' \cos^2\theta\sin^3\theta_z).
    '''
    def one(i: Int) {surface_tilt, surface_azimuth, dhi, ghi, solar_zenith,
                      solar_azimuth, poa_sky_diffuse}:
        # zenith angle with respect to panel normal.
        var aoi = _aoi_parts(
            surface_tilt.unsafe_load(i), surface_azimuth.unsafe_load(i), solar_zenith.unsafe_load(i), solar_azimuth.unsafe_load(i)
        )
        var cos_tt = maximum(aoi.projection, 0)  # GH 526

        # silence warning from 0 / 0
        var F = 1 - ((dhi.unsafe_load(i) / ghi.unsafe_load(i)) ** 2)

        # fails with single point input
        if isnan(F):
            F = 0

        var term1 = 0.5 * (1 + aoi.cos_tilt)
        var term2 = 1 + F * (sind_(0.5 * surface_tilt.unsafe_load(i)) ** 3)
        var term3 = 1 + F * (cos_tt ** 2) * (sind_(solar_zenith.unsafe_load(i)) ** 3)

        poa_sky_diffuse.unsafe_store(i, dhi.unsafe_load(i) * term1 * term2 * term3)

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def haydavies(
    surface_tilt: F64, surface_azimuth: F64, dhi: F64, dni: F64, dni_extra: F64,
    solar_zenith: F64, solar_azimuth: F64, projection_ratio: F64,
    poa_sky_diffuse: F64, poa_isotropic: F64, poa_circumsolar: F64, n: Int,
    use_projection_ratio: Int
):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using the
    Hay-Davies (1980) model.
    '''
    def one(i: Int) {surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                      solar_zenith, solar_azimuth, projection_ratio,
                      poa_isotropic, poa_circumsolar, poa_sky_diffuse,
                      use_projection_ratio}:
        var aoi = _aoi_parts(
            surface_tilt.unsafe_load(i), surface_azimuth.unsafe_load(i), solar_zenith.unsafe_load(i), solar_azimuth.unsafe_load(i)
        )
        # if necessary, calculate ratio of titled and horizontal beam irradiance
        var Rb = 0.0
        if use_projection_ratio == 0:
            var cos_tt = maximum(aoi.projection, 0)  # GH 526
            Rb = cos_tt / maximum(aoi.cos_zen, 0.01745)  # GH 432
        else:
            Rb = projection_ratio.unsafe_load(i)

        # Anisotropy Index
        var AI = dni.unsafe_load(i) / dni_extra.unsafe_load(i)

        # these are the () and [] sub-terms of the second term of eqn 7
        var term1 = 1 - AI
        var term2 = 0.5 * (1 + aoi.cos_tilt)

        poa_isotropic.unsafe_store(i, maximum(dhi.unsafe_load(i) * term1 * term2, 0))
        poa_circumsolar.unsafe_store(i, maximum(dhi.unsafe_load(i) * (AI * Rb), 0))
        poa_sky_diffuse.unsafe_store(i, poa_isotropic.unsafe_load(i) + poa_circumsolar.unsafe_load(i))

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())



def reindl(
    surface_tilt: F64, surface_azimuth: F64, dhi: F64, dni: F64, ghi: F64,
    dni_extra: F64, solar_zenith: F64, solar_azimuth: F64, poa_sky_diffuse: F64,
    poa_isotropic: F64, poa_circumsolar: F64, poa_horizon: F64, n: Int
):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using the
    Reindl (1990) model.
    '''
    def one(i: Int) {surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
                      solar_zenith, solar_azimuth, poa_isotropic,
                      poa_circumsolar, poa_horizon, poa_sky_diffuse}:
        var aoi = _aoi_parts(
            surface_tilt.unsafe_load(i), surface_azimuth.unsafe_load(i), solar_zenith.unsafe_load(i), solar_azimuth.unsafe_load(i)
        )
        # Rb
        var cos_tt = maximum(aoi.projection, 0)  # GH 526
        var cos_solar_zenith = aoi.cos_zen
        var Rb = cos_tt / maximum(cos_solar_zenith, 0.01745)  # GH 432

        # Anisotropy Index
        var AI = dni.unsafe_load(i) / dni_extra.unsafe_load(i)

        # DNI projected onto horizontal
        var HB = dni.unsafe_load(i) * cos_solar_zenith
        HB = maximum(HB, 0)

        var SVF = (1 + aoi.cos_tilt) / 2

        var hb_to_ghi = 0.0
        if ghi.unsafe_load(i) != 0:
            hb_to_ghi = HB / ghi.unsafe_load(i)
        var h = sqrt(hb_to_ghi) * (sind_(surface_tilt.unsafe_load(i) / 2) ** 3)

        var term1 = (1 - AI) * SVF
        var term2 = AI * Rb
        var term3 = term1 * h

        poa_isotropic.unsafe_store(i, dhi.unsafe_load(i) * term1)
        poa_circumsolar.unsafe_store(i, dhi.unsafe_load(i) * term2)
        poa_horizon.unsafe_store(i, dhi.unsafe_load(i) * term3)
        poa_sky_diffuse.unsafe_store(i, dhi.unsafe_load(i) * (term1 + term2 + term3))

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def king(
    surface_tilt: F64, dhi: F64, ghi: F64, solar_zenith: F64,
    poa_sky_diffuse: F64, n: Int
):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using the
    King model.
    '''
    def one(i: Int) {surface_tilt, dhi, ghi, solar_zenith, poa_sky_diffuse}:
        var cos_tilt = cosd_(surface_tilt.unsafe_load(i))
        var sky = (
            dhi.unsafe_load(i) * (1 + cos_tilt) / 2 + ghi.unsafe_load(i) *
            (0.012 * solar_zenith.unsafe_load(i) - 0.04) *
            (1 - cos_tilt) / 2
        )
        poa_sky_diffuse.unsafe_store(i, maximum(sky, 0))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def perez(
    surface_tilt: F64, surface_azimuth: F64, dhi: F64, dni: F64, dni_extra: F64,
    solar_zenith: F64, solar_azimuth: F64, airmass: F64, f1c: F64, f2c: F64,
    poa_sky_diffuse: F64, poa_isotropic: F64, poa_circumsolar: F64,
    poa_horizon: F64, n: Int
):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using the
    Perez model.
    '''
    var kappa = 1.041  # for solar_zenith in radians
    def one(i: Int) {surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                      solar_zenith, solar_azimuth, airmass, f1c, f2c,
                      poa_isotropic, poa_circumsolar, poa_horizon,
                      poa_sky_diffuse, kappa}:
        var z = radians_(solar_zenith.unsafe_load(i))  # convert to radians

        # delta is the sky's "brightness"
        var delta = dhi.unsafe_load(i) * airmass.unsafe_load(i) / dni_extra.unsafe_load(i)

        # epsilon is the sky's "clearness"
        var eps = ((dhi.unsafe_load(i) + dni.unsafe_load(i)) / dhi.unsafe_load(i) + kappa * (z**3)) / (1 + kappa * (z**3))

        # Perez et al define clearness bins according to the following
        # rules. 1 = overcast ... 8 = clear (these names really only make
        # sense for small zenith angles, but...) these values will
        # eventually be used as indicies for coeffecient look ups
        # `ebin` is upstream's post-decrement value, so -1 flags an invalid
        # eps; upstream's NaN coefficient row is index -1 in numpy and index 0
        # in the table this kernel reads.
        var ebin = -1
        if not isnan(eps):
            ebin = 0
            ebin += 1 if eps >= 0. else 0
            ebin += 1 if eps >= 1.065 else 0
            ebin += 1 if eps >= 1.23 else 0
            ebin += 1 if eps >= 1.5 else 0
            ebin += 1 if eps >= 1.95 else 0
            ebin += 1 if eps >= 2.8 else 0
            ebin += 1 if eps >= 4.5 else 0
            ebin += 1 if eps >= 6.2 else 0
            ebin -= 1

        var r = ebin + 1
        var F1 = (f1c.unsafe_load(r * 3) + f1c.unsafe_load(r * 3 + 1) * delta + f1c.unsafe_load(r * 3 + 2) * z)
        F1 = maximum(F1, 0)

        var F2 = (f2c.unsafe_load(r * 3) + f2c.unsafe_load(r * 3 + 1) * delta + f2c.unsafe_load(r * 3 + 2) * z)

        var aoi = _aoi_parts(
            surface_tilt.unsafe_load(i), surface_azimuth.unsafe_load(i), solar_zenith.unsafe_load(i), solar_azimuth.unsafe_load(i)
        )
        var A = maximum(aoi.projection, 0)

        var B = maximum(aoi.cos_zen, cosd_(85))

        # Calculate Diffuse POA from sky dome
        var term1 = 0.5 * (1 - F1) * (1 + aoi.cos_tilt)
        var term2 = F1 * A / B
        var term3 = F2 * sind_(surface_tilt.unsafe_load(i))

        poa_isotropic.unsafe_store(i, dhi.unsafe_load(i) * term1)
        poa_circumsolar.unsafe_store(i, dhi.unsafe_load(i) * term2)
        poa_horizon.unsafe_store(i, dhi.unsafe_load(i) * term3)
        var sky = maximum(dhi.unsafe_load(i) * (term1 + term2 + term3), 0)
        if isnan(airmass.unsafe_load(i)):
            sky = 0.0
        poa_sky_diffuse.unsafe_store(i, sky)

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def clearsky_index(
    ghi: F64, ghi_clear: F64, clearsky_index: F64, n: Int,
    max_clearsky_index: Float64
):
    """
    Calculate the clearsky index.
    """
    # No transcendental here, so the serial pass is bound by the loads and the
    # divide, and it already runs in about 3 ms per million elements; the
    # thread launch costs more than that, so this one stays serial. A
    # branch-free rewrite of the three selects below measured identical to
    # this form, so it is not worth carrying.
    for i in range(n):
        var k = ghi.unsafe_load(i) / ghi_clear.unsafe_load(i)
        # set +inf, -inf, and nans to zero
        if not isfinite(k):
            k = 0.0
        # but preserve nans in the input arrays
        if not isfinite(ghi.unsafe_load(i)) or not isfinite(ghi_clear.unsafe_load(i)):
            k = nan_()

        k = maximum(k, 0)
        clearsky_index.unsafe_store(i, minimum(k, max_clearsky_index))


def _clearness_index(
    ghi: Float64, solar_zenith: Float64, extra_radiation: Float64,
    min_cos_zenith: Float64, max_clearness_index: Float64
) -> Float64:
    """`clearness_index` for one timestamp."""
    return _clearness_index_cos(cosd_(solar_zenith), ghi, extra_radiation,
                                min_cos_zenith, max_clearness_index)


def _clearness_index_cos(
    cos_zenith: Float64, ghi: Float64, extra_radiation: Float64,
    min_cos_zenith: Float64, max_clearness_index: Float64
) -> Float64:
    """`_clearness_index` with `cosd(solar_zenith)` already evaluated.

    The decomposition models need that cosine again to turn GHI - DHI back
    into DNI, so they hand it in rather than paying for it twice.
    """
    var I0h = extra_radiation * maximum(cos_zenith, min_cos_zenith)
    var kt = ghi / I0h
    kt = maximum(kt, 0)
    return minimum(kt, max_clearness_index)


def clearness_index(
    ghi: F64, solar_zenith: F64, extra_radiation: F64, kt: F64, n: Int,
    min_cos_zenith: Float64, max_clearness_index: Float64
):
    """
    Calculate the clearness index.
    """
    def one(i: Int) {ghi, solar_zenith, extra_radiation, kt, min_cos_zenith,
                      max_clearness_index}:
        kt.unsafe_store(i, _clearness_index(
            ghi.unsafe_load(i), solar_zenith.unsafe_load(i),
            extra_radiation.unsafe_load(i), min_cos_zenith, max_clearness_index
        ))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def _clearness_index_zenith_independent(
    clearness_index: Float64, airmass: Float64, max_clearness_index: Float64
) -> Float64:
    # Perez eqn 1
    var kt_prime = clearness_index / _kt_kt_prime_factor(airmass)
    kt_prime = maximum(kt_prime, 0)
    return minimum(kt_prime, max_clearness_index)


def clearness_index_zenith_independent(
    clearness_index: F64, airmass: F64, kt_prime: F64, n: Int,
    max_clearness_index: Float64
):
    """
    Calculate the zenith angle independent clearness index.
    """
    def one(i: Int) {clearness_index, airmass, kt_prime, max_clearness_index}:
        kt_prime.unsafe_store(i, _clearness_index_zenith_independent(
            clearness_index.unsafe_load(i), airmass.unsafe_load(i), max_clearness_index
        ))

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def _kt_kt_prime_factor(airmass: Float64) -> Float64:
    """
    Calculate the conversion factor between kt and kt prime.
    Function is useful because DIRINT and GTI-DIRINT both use this.
    """
    # consider adding
    # airmass = np.maximum(airmass, 12)  # GH 450
    return 1.031 * exp(-1.4 / (0.9 + 9.4 / airmass)) + 0.1


struct DiscKn:
    var Kn: Float64
    var am: Float64

    def __init__(out self, Kn: Float64, am: Float64):
        self.Kn = Kn
        self.am = am


def _disc_kn(clearness_index: Float64, airmass: Float64, max_airmass: Float64) -> DiscKn:
    """
    Calculate Kn for `disc`
    """
    # short names for equations
    var kt = clearness_index
    var am = airmass

    am = minimum(am, max_airmass)  # GH 450

    var is_cloudy = kt <= 0.6
    # Use Horner's method to compute polynomials efficiently
    var a: Float64
    var b: Float64
    var c: Float64
    if is_cloudy:
        a = 0.512 + kt * (-1.56 + kt * (2.286 - 2.222 * kt))
        b = 0.37 + 0.962 * kt
        c = -0.28 + kt * (0.932 - 2.048 * kt)
    else:
        a = -5.743 + kt * (21.77 + kt * (-27.49 + 11.56 * kt))
        b = 41.4 + kt * (-118.5 + kt * (66.05 + 31.9 * kt))
        c = -47.01 + kt * (184.2 + kt * (-222.0 + 73.81 * kt))

    var delta_kn = a + b * exp(c * am)

    var Knc = 0.866 + am * (-0.122 + am * (0.0121 + am * (-0.000653 + 1.4e-05 * am)))
    return DiscKn(Knc - delta_kn, am)


struct Disc:
    var dni: Float64
    var kt: Float64
    var airmass: Float64

    def __init__(out self, dni: Float64, kt: Float64, airmass: Float64):
        self.dni = dni
        self.kt = kt
        self.airmass = airmass


def _disc(
    ghi: Float64, solar_zenith: Float64, doy: Float64, pressure: Float64,
    min_cos_zenith: Float64, max_zenith: Float64, max_airmass: Float64,
    use_pressure: Int
) -> Disc:
    """`disc` for one timestamp."""
    # this is the I0 calculation from the reference
    # SSC uses solar constant = 1367.0 (checked 2018 08 15)
    var I0 = _get_extra_radiation(doy, 1370., EXTRA_RADIATION_SPENCER)

    var kt = _clearness_index(ghi, solar_zenith, I0, min_cos_zenith, 1)

    var am = relative_airmass_(solar_zenith, AIRMASS_KASTEN1966)
    if use_pressure != 0:
        am = absolute_airmass_(am, pressure)

    var kn = _disc_kn(kt, am, max_airmass)
    am = kn.am
    var dni = kn.Kn * I0

    var bad_values = (solar_zenith > max_zenith) or (ghi < 0) or (dni < 0)
    if bad_values:
        dni = 0.0

    return Disc(dni, kt, am)


def disc(
    ghi: F64, solar_zenith: F64, doy: F64, dni: F64, kt: F64, airmass: F64, n: Int,
    pressure: Float64, min_cos_zenith: Float64, max_zenith: Float64,
    max_airmass: Float64, use_pressure: Int
):
    """
    Determine DNI from GHI using the DISC model.
    """
    def one(i: Int) {ghi, solar_zenith, doy, dni, kt, airmass, pressure,
                      min_cos_zenith, max_zenith, max_airmass, use_pressure}:
        var out = _disc(
            ghi.unsafe_load(i), solar_zenith.unsafe_load(i), doy.unsafe_load(i), pressure, min_cos_zenith,
            max_zenith, max_airmass, use_pressure
        )
        dni.unsafe_store(i, out.dni)
        kt.unsafe_store(i, out.kt)
        airmass.unsafe_store(i, out.airmass)

    if n < PAR_MIN_HEAVY:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def _dirint_bins(
    kt_prime: Float64, zenith: Float64, w: Float64, delta_kt_prime: Float64
) -> Tuple[Int, Int, Int, Int]:
    """
    Determine the bins for the DIRINT coefficients.

    The following bin assignments use MATLAB's 1-indexing; the coefficient
    lookup subtracts 1 to conform to Python's 0-indexing.
    """
    var kt_prime_bin = 0
    var zenith_bin = 0
    var w_bin = 0
    var delta_kt_prime_bin = 0

    # Create kt_prime bins
    if (kt_prime >= 0) and (kt_prime < 0.24):
        kt_prime_bin = 1
    if (kt_prime >= 0.24) and (kt_prime < 0.4):
        kt_prime_bin = 2
    if (kt_prime >= 0.4) and (kt_prime < 0.56):
        kt_prime_bin = 3
    if (kt_prime >= 0.56) and (kt_prime < 0.7):
        kt_prime_bin = 4
    if (kt_prime >= 0.7) and (kt_prime < 0.8):
        kt_prime_bin = 5
    if (kt_prime >= 0.8) and (kt_prime <= 1):
        kt_prime_bin = 6

    # Create zenith angle bins
    if (zenith >= 0) and (zenith < 25):
        zenith_bin = 1
    if (zenith >= 25) and (zenith < 40):
        zenith_bin = 2
    if (zenith >= 40) and (zenith < 55):
        zenith_bin = 3
    if (zenith >= 55) and (zenith < 70):
        zenith_bin = 4
    if (zenith >= 70) and (zenith < 80):
        zenith_bin = 5
    if zenith >= 80:
        zenith_bin = 6

    # Create the bins for w based on dew point temperature
    if (w >= 0) and (w < 1):
        w_bin = 1
    if (w >= 1) and (w < 2):
        w_bin = 2
    if (w >= 2) and (w < 3):
        w_bin = 3
    if w >= 3:
        w_bin = 4
    if w == -1:
        w_bin = 5

    # Create delta_kt_prime binning.
    if (delta_kt_prime >= 0) and (delta_kt_prime < 0.015):
        delta_kt_prime_bin = 1
    if (delta_kt_prime >= 0.015) and (delta_kt_prime < 0.035):
        delta_kt_prime_bin = 2
    if (delta_kt_prime >= 0.035) and (delta_kt_prime < 0.07):
        delta_kt_prime_bin = 3
    if (delta_kt_prime >= 0.07) and (delta_kt_prime < 0.15):
        delta_kt_prime_bin = 4
    if (delta_kt_prime >= 0.15) and (delta_kt_prime < 0.3):
        delta_kt_prime_bin = 5
    if (delta_kt_prime >= 0.3) and (delta_kt_prime <= 1):
        delta_kt_prime_bin = 6
    if delta_kt_prime == -1:
        delta_kt_prime_bin = 7

    return (kt_prime_bin, zenith_bin, w_bin, delta_kt_prime_bin)


def _dirint_coeffs(
    coeffs: F64, kt_prime: Float64, solar_zenith: Float64, w: Float64,
    delta_kt_prime: Float64
) -> Float64:
    """`_dirint_coeffs` for one timestamp."""
    var bins = _dirint_bins(kt_prime, solar_zenith, w, delta_kt_prime)
    var kt_prime_bin = bins[0]
    var zenith_bin = bins[1]
    var w_bin = bins[2]
    var delta_kt_prime_bin = bins[3]

    # convert unassigned bins to nan. This has to come before the lookup:
    # a bin of 0 would make the index below negative, and the buffer has no
    # element in front of it to read. A NaN in any operand leaves every
    # comparison false and every bin at 0, so this is the ordinary path for a
    # NaN input rather than a corner case.
    if (kt_prime_bin == 0) or (zenith_bin == 0) or (w_bin == 0) or (delta_kt_prime_bin == 0):
        return nan_()

    # get the coefficients
    # subtract 1 to account for difference between MATLAB-style bin
    # assignment and Python-style array lookup.
    return coeffs.unsafe_load(
        ((kt_prime_bin - 1) * 6 + (zenith_bin - 1)) * 7 * 5 +
        (delta_kt_prime_bin - 1) * 5 + (w_bin - 1)
    )


def _delta_kt_prime(
    kt_prime: F64, delta_kt_prime: F64, n: Int, use_delta_kt_prime: Int
):
    """`_delta_kt_prime_dirint` over the whole series.

    Row-wise mean of neighbour abs-differences; pandas skips NaN so this
    covers 0/1/2 valid neighbours (Perez eqn 2 interior, eqn 3 boundary/gap).
    """
    for i in range(n):
        if use_delta_kt_prime == 0:
            # do not change unless also modifying _dirint_bins
            delta_kt_prime.unsafe_store(i, -1)
        else:
            var acc = 0.0
            var count = 0.0
            if i + 1 < n and not isnan(kt_prime.unsafe_load(i + 1)):
                acc += abs(kt_prime.unsafe_load(i) - kt_prime.unsafe_load(i + 1))
                count += 1
            if i > 0 and not isnan(kt_prime.unsafe_load(i - 1)):
                acc += abs(kt_prime.unsafe_load(i) - kt_prime.unsafe_load(i - 1))
                count += 1
            delta_kt_prime.unsafe_store(i, acc / count)


def _temp_dew_dirint(temp_dew: F64, w: F64, n: Int, use_temp_dew: Int):
    """`_temp_dew_dirint` over the whole series (Perez eqn 4)."""
    for i in range(n):
        if use_temp_dew == 0:
            # do not change unless also modifying _dirint_bins
            w.unsafe_store(i, -1)
        else:
            w.unsafe_store(i, exp(0.07 * temp_dew.unsafe_load(i) - 0.075))


def dirint(
    ghi: F64, solar_zenith: F64, doy: F64, delta_kt_prime: F64, temp_dew: F64,
    coeffs: F64, dni: F64, n: Int, pressure: Float64, max_airmass: Float64,
    use_pressure: Int, use_delta_kt_prime: Int, use_temp_dew: Int,
    min_cos_zenith: Float64, max_zenith: Float64, scratch: F64
):
    """
    Determine DNI from GHI using the DIRINT model.
    """
    # scratch is 3n: the DISC airmass, kt' and the precipitable water
    var airmass = scratch
    var kt_prime = scratch.unsafe_offset(n)
    var w = scratch.unsafe_offset(2 * n)

    def disc_pass(i: Int) {ghi, solar_zenith, doy, dni, airmass, kt_prime,
                           pressure, min_cos_zenith, max_zenith, max_airmass,
                           use_pressure}:
        var out = _disc(
            ghi.unsafe_load(i), solar_zenith.unsafe_load(i), doy.unsafe_load(i), pressure, min_cos_zenith,
            max_zenith, max_airmass, use_pressure
        )
        # Perez eqn 5 multiplies DISC's DNI by the coefficients below
        dni.unsafe_store(i, out.dni)
        airmass.unsafe_store(i, out.airmass)
        kt_prime.unsafe_store(i, _clearness_index_zenith_independent(out.kt, out.airmass, 1))

    if n < PAR_MIN_HEAVY:
        for i in range(n):
            disc_pass(i)
    else:
        parallelize(disc_pass, n, par_workers())

    _delta_kt_prime(kt_prime, delta_kt_prime, n, use_delta_kt_prime)
    _temp_dew_dirint(temp_dew, w, n, use_temp_dew)

    # The coefficient lookup is a table read and a multiply, so it stays
    # serial: a second thread launch would cost more than the pass does.
    for i in range(n):
        dni.unsafe_store(i, dni.unsafe_load(i) * _dirint_coeffs(
            coeffs, kt_prime.unsafe_load(i), solar_zenith.unsafe_load(i), w.unsafe_load(i),
            delta_kt_prime.unsafe_load(i)
        ))


def dirindex(
    dni_clear: F64, dni_dirint: F64, dni_dirint_clearsky: F64,
    dni_dirindex: F64, n: Int
):
    """
    The DIRINDEX combination of two DIRINT runs against a clear-sky DNI.
    """
    # Memory-bound like `clearsky_index`, so serial at every size.
    for i in range(n):
        var dni = dni_clear.unsafe_load(i) * dni_dirint.unsafe_load(i) / dni_dirint_clearsky.unsafe_load(i)
        dni_dirindex.unsafe_store(i, 0. if dni < 0 else dni)


def erbs(
    ghi: F64, zenith: F64, dni_extra: F64, dni: F64, dhi: F64, kt: F64, n: Int,
    min_cos_zenith: Float64, max_zenith: Float64
):
    r"""
    Estimate DNI and DHI from GHI using the Erbs model.
    """
    def one(i: Int) {ghi, zenith, dni_extra, dni, dhi, kt, min_cos_zenith, max_zenith}:
        var cos_zen = cosd_(zenith.unsafe_load(i))
        var k = _clearness_index_cos(cos_zen, ghi.unsafe_load(i), dni_extra.unsafe_load(i), min_cos_zenith, 1)

        # For Kt <= 0.22, set the diffuse fraction
        var df = 1 - 0.09 * k

        # For Kt > 0.22 and Kt <= 0.8, set the diffuse fraction
        if (k > 0.22) and (k <= 0.8):
            df = 0.9511 - 0.1604 * k + 4.388 * k**2 - 16.638 * k**3 + 12.336 * k**4

        # For Kt > 0.8, set the diffuse fraction
        if k > 0.8:
            df = 0.165

        dhi.unsafe_store(i, df * ghi.unsafe_load(i))

        dni.unsafe_store(i, (ghi.unsafe_load(i) - dhi.unsafe_load(i)) / cos_zen)
        var bad_values = (zenith.unsafe_load(i) > max_zenith) or (ghi.unsafe_load(i) < 0) or (dni.unsafe_load(i) < 0)
        if bad_values:
            dni.unsafe_store(i, 0)
        # ensure that closure relationship remains valid
        if bad_values:
            dhi.unsafe_store(i, ghi.unsafe_load(i))
        kt.unsafe_store(i, k)

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def orgill_hollands(
    ghi: F64, zenith: F64, dni_extra: F64, dni: F64, dhi: F64, kt: F64, n: Int,
    min_cos_zenith: Float64, max_zenith: Float64
):
    """Estimate DNI and DHI from GHI using the Orgill and Hollands model."""
    def one(i: Int) {ghi, zenith, dni_extra, dni, dhi, kt, min_cos_zenith, max_zenith}:
        var cos_zen = cosd_(zenith.unsafe_load(i))
        var k = _clearness_index_cos(cos_zen, ghi.unsafe_load(i), dni_extra.unsafe_load(i), min_cos_zenith, 1)

        # For Kt < 0.35, set the diffuse fraction
        var df = 1 - 0.249 * k

        # For Kt >= 0.35 and Kt <= 0.75, set the diffuse fraction
        if (k >= 0.35) and (k <= 0.75):
            df = 1.557 - 1.84 * k

        # For Kt > 0.75, set the diffuse fraction
        if k > 0.75:
            df = 0.177

        dhi.unsafe_store(i, df * ghi.unsafe_load(i))

        dni.unsafe_store(i, (ghi.unsafe_load(i) - dhi.unsafe_load(i)) / cos_zen)
        var bad_values = (zenith.unsafe_load(i) > max_zenith) or (ghi.unsafe_load(i) < 0) or (dni.unsafe_load(i) < 0)
        if bad_values:
            dni.unsafe_store(i, 0)
        # ensure that closure relationship remains valid
        if bad_values:
            dhi.unsafe_store(i, ghi.unsafe_load(i))
        kt.unsafe_store(i, k)

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def boland(
    ghi: F64, solar_zenith: F64, dni_extra: F64, dni: F64, dhi: F64, kt: F64,
    n: Int, a_coeff: Float64, b_coeff: Float64, min_cos_zenith: Float64,
    max_zenith: Float64
):
    r"""
    Estimate DNI and DHI from GHI using the Boland clearness index model.
    """
    def one(i: Int) {ghi, solar_zenith, dni_extra, dni, dhi, kt, a_coeff,
                      b_coeff, min_cos_zenith, max_zenith}:
        var cos_zen = cosd_(solar_zenith.unsafe_load(i))
        var k = _clearness_index_cos(
            cos_zen, ghi.unsafe_load(i), dni_extra.unsafe_load(i), min_cos_zenith, 1
        )

        # Boland equation
        var df = 1.0 / (1.0 + exp(a_coeff * (k - b_coeff)))
        # NOTE: [2] has different coefficients, for different time intervals
        # 15-min: df = 1 / (1 + exp(8.645 * (kt - 0.613)))
        # 1-hour: df = 1 / (1 + exp(7.997 * (kt - 0.586)))

        dhi.unsafe_store(i, df * ghi.unsafe_load(i))

        dni.unsafe_store(i, (ghi.unsafe_load(i) - dhi.unsafe_load(i)) / cos_zen)
        var bad_values = (solar_zenith.unsafe_load(i) > max_zenith) or (ghi.unsafe_load(i) < 0) or (dni.unsafe_load(i) < 0)
        if bad_values:
            dni.unsafe_store(i, 0)
        # ensure that closure relationship remains valid
        if bad_values:
            dhi.unsafe_store(i, ghi.unsafe_load(i))
        kt.unsafe_store(i, k)

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def dni(
    ghi: F64, dhi: F64, zenith: F64, dni_clear: F64, dni_out: F64, n: Int,
    dni_clear_tolerance: Float64, use_dni_clear: Int,
    zenith_threshold_for_zero_dni: Float64, zenith_threshold_for_clearsky_limit: Float64
):
    """
    Calculate DNI from GHI and DHI.
    """
    def one(i: Int) {ghi, dhi, zenith, dni_clear, dni_out, dni_clear_tolerance,
                      use_dni_clear, zenith_threshold_for_zero_dni,
                      zenith_threshold_for_clearsky_limit}:
        # calculate DNI
        var d = (ghi.unsafe_load(i) - dhi.unsafe_load(i)) / cosd_(zenith.unsafe_load(i))

        # cutoff negative values
        if d < 0:
            d = nan_()

        # set non-zero DNI values for zenith angles >=
        # zenith_threshold_for_zero_dni to NaN
        if (zenith.unsafe_load(i) >= zenith_threshold_for_zero_dni) and (d != 0):
            d = nan_()

        # correct DNI values for zenith angles greater or equal to the
        # zenith_threshold_for_clearsky_limit and smaller than the
        # upper_cutoff_zenith that are greater than the clearsky DNI (times
        # clearsky_tolerance)
        if use_dni_clear != 0:
            var max_dni = dni_clear.unsafe_load(i) * dni_clear_tolerance
            if (zenith.unsafe_load(i) >= zenith_threshold_for_clearsky_limit) and (zenith.unsafe_load(i) < zenith_threshold_for_zero_dni) and (d > max_dni):
                d = max_dni
        dni_out.unsafe_store(i, d)

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def louche(
    ghi: F64, solar_zenith: F64, dni_extra: F64, dni: F64, dhi: F64, kt: F64,
    n: Int, max_zenith: Float64
):
    """
    Determine DNI and DHI from GHI using the Louche model.
    """
    def one(i: Int) {ghi, solar_zenith, dni_extra, dni, dhi, kt, max_zenith}:
        var cos_zen = cosd_(solar_zenith.unsafe_load(i))
        var Kt = _clearness_index_cos(cos_zen, ghi.unsafe_load(i), dni_extra.unsafe_load(i), 0.065, 2.0)

        var kb = -10.627 * Kt**5 + 15.307 * Kt**4 - 5.205 * Kt**3 + 0.994 * Kt**2 - 0.059 * Kt + 0.002
        dni.unsafe_store(i, kb * dni_extra.unsafe_load(i))
        dhi.unsafe_store(i, ghi.unsafe_load(i) - dni.unsafe_load(i) * cos_zen)

        var bad_values = (solar_zenith.unsafe_load(i) > max_zenith) or (ghi.unsafe_load(i) < 0) or (dni.unsafe_load(i) < 0)
        if bad_values:
            dni.unsafe_store(i, 0)
        # ensure that closure relationship remains valid
        if bad_values:
            dhi.unsafe_store(i, ghi.unsafe_load(i))
        kt.unsafe_store(i, Kt)

    if n < PAR_MIN_MEDIUM:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def campbell_norman(
    zenith: F64, ghi: F64, dni: F64, dhi: F64, n: Int, transmittance: Float64,
    pressure: Float64, dni_extra: Float64
):
    """
    Determine DNI, DHI, GHI from extraterrestrial flux, transmittance,
    and atmospheric pressure.
    """
    def one(i: Int) {zenith, ghi, dni, dhi, transmittance, pressure, dni_extra}:
        var z = zenith.unsafe_load(i)
        var airmass = relative_airmass_(z, AIRMASS_SIMPLE)
        airmass = absolute_airmass_(airmass, pressure)
        var tau_am = pow_(transmittance, airmass)
        var cos_zen = cosd_(z)
        var dni_v = dni_extra * tau_am
        var dhi_v = 0.3 * (1.0 - tau_am) * dni_extra * cos_zen
        ghi.unsafe_store(i, dhi_v + dni_v * cos_zen)
        dni.unsafe_store(i, dni_v)
        dhi.unsafe_store(i, dhi_v)

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def diffuse_par_spitters(
    daily_solar_zenith: F64, global_diffuse_fraction: F64,
    par_diffuse_fraction: F64, n: Int
):
    r"""
    Derive daily diffuse fraction of Photosynthetically Active Radiation (PAR)
    from daily average solar zenith and diffuse fraction of daily insolation.

    The relationship is based on the work of Spitters et al. (1986).
    """
    # cosd(90-x) = sind(x) and 90-solar_elevation = solar_zenith
    def one(i: Int) {daily_solar_zenith, global_diffuse_fraction,
                      par_diffuse_fraction}:
        var zen = daily_solar_zenith.unsafe_load(i)
        var kd = global_diffuse_fraction.unsafe_load(i)
        var cosd_solar_zenith = cosd_(zen)
        var cosd_solar_elevation = sind_(zen)
        par_diffuse_fraction.unsafe_store(
            i,
            (1 + 0.3 * (1 - kd**2)) * kd /
            (1 + (1 - kd**2) * cosd_solar_zenith**2 * cosd_solar_elevation**3),
        )

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())


def poa_components(
    aoi: F64, dni: F64, poa_sky_diffuse: F64, poa_ground_diffuse: F64,
    poa_direct: F64, poa_diffuse: F64, poa_global: F64, n: Int
):
    r"""
    Determine the total plane of array irradiance from the components already
    computed upstream.

    ``poa_direct``, ``poa_diffuse`` and ``poa_global`` are elementwise functions
    of the same four inputs, so one pass writes all three; upstream walks them
    as three whole-array numpy expressions, each reading the buffers again.
    """
    def one(i: Int) {aoi, dni, poa_sky_diffuse, poa_ground_diffuse,
                      poa_direct, poa_diffuse, poa_global}:
        var direct = maximum(dni.unsafe_load(i) * cosd_(aoi.unsafe_load(i)), 0)
        var diffuse = poa_sky_diffuse.unsafe_load(i) + poa_ground_diffuse.unsafe_load(i)
        poa_direct.unsafe_store(i, direct)
        poa_diffuse.unsafe_store(i, diffuse)
        poa_global.unsafe_store(i, direct + diffuse)

    if n < PAR_MIN_CHEAP:
        for i in range(n):
            one(i)
        return
    parallelize(one, n, par_workers())
