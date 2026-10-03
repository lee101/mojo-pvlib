"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric. Callers own all
memory, including scratch — nothing here allocates, so nothing here can leak.
Each wrapper is a straight transcription of the kernel's own signature; the
upstream logic stays in `src/*.mojo`.
"""

from types import F64, p

from tools import cosd_, sind_, tand_, asind_, acosd_, atand_
from solarposition import (
    _calculate_simple_day_angle, equation_of_time_spencer71,
    equation_of_time_pvcdrom, declination_spencer71, declination_cooper69,
    solar_azimuth_analytical, solar_zenith_analytical, hour_angle, sunset_angle,
)
from atmosphere import (
    pres2alt, alt2pres, get_absolute_airmass, get_relative_airmass, gueymard94_pw,
    bird_hulstrom80_aod_bb, kasten96_lt, rh_from_tdew, tdew_from_rh,
    angstrom_aod_at_lambda, angstrom_alpha, windspeed_powerlaw,
)
from irradiance import (
    get_extra_radiation, aoi_projection, aoi, beam_component,
    get_ground_diffuse, isotropic, klucher, haydavies, reindl, king, perez,
    clearsky_index, clearness_index, clearness_index_zenith_independent,
    disc, dirint, dirindex, erbs, orgill_hollands, boland, dni, louche,
    campbell_norman, diffuse_par_spitters, poa_components,
    EXTRA_RADIATION_ASCE, EXTRA_RADIATION_SPENCER,
)
from spa import solar_position_loop, calculate_deltat
from spa_tables import spa_tables
from irradiance_tables import dirint_coeffs_table, perez_coefficients


# ------------------------------------------------------------------- tools
@export("mpv_cosd")
def mpv_cosd(angle: Int, res: Int, n: Int) abi("C"):
    cosd_(p(angle), p(res), n)


@export("mpv_sind")
def mpv_sind(angle: Int, res: Int, n: Int) abi("C"):
    sind_(p(angle), p(res), n)


@export("mpv_tand")
def mpv_tand(angle: Int, res: Int, n: Int) abi("C"):
    tand_(p(angle), p(res), n)


@export("mpv_asind")
def mpv_asind(number: Int, res: Int, n: Int) abi("C"):
    asind_(p(number), p(res), n)


@export("mpv_acosd")
def mpv_acosd(number: Int, res: Int, n: Int) abi("C"):
    acosd_(p(number), p(res), n)


@export("mpv_atand")
def mpv_atand(number: Int, res: Int, n: Int) abi("C"):
    atand_(p(number), p(res), n)


# --------------------------------------------------------------------- spa
@export("mpv_sp_tables")
def mpv_sp_tables(dst: Int) abi("C"):
    spa_tables(p(dst))


@export("mpv_solar_position_loop")
def mpv_solar_position_loop(
    unixtime: Int,
    delta_t: Int,
    n: Int,
    lat: Float64,
    lon: Float64,
    elev: Float64,
    pressure: Float64,
    temp: Float64,
    atmos_refract: Float64,
    sst: Int,
    esd: Int,
    theta: Int,
    theta0: Int,
    e: Int,
    e0: Int,
    phi: Int,
    eot: Int,
    tb: Int,
) abi("C"):
    solar_position_loop(
        p(unixtime), p(delta_t), n, lat, lon, elev, pressure, temp,
        atmos_refract, sst, esd, p(theta), p(theta0), p(e), p(e0), p(phi),
        p(eot), p(tb)
    )


@export("mpv_calculate_deltat")
def mpv_calculate_deltat(year: Int, month: Int, deltat: Int, n: Int) abi("C"):
    calculate_deltat(p(year), p(month), p(deltat), n)


# ---------------------------------------------------------- solarposition
@export("mpv_calculate_simple_day_angle")
def mpv_calculate_simple_day_angle(
    dayofyear: Int, offset: Float64, day_angle: Int, n: Int
) abi("C"):
    _calculate_simple_day_angle(p(dayofyear), offset, p(day_angle), n)


@export("mpv_equation_of_time_spencer71")
def mpv_equation_of_time_spencer71(
    dayofyear: Int, eot: Int, n: Int, day_angle: Int
) abi("C"):
    equation_of_time_spencer71(p(dayofyear), p(eot), n, p(day_angle))


@export("mpv_equation_of_time_pvcdrom")
def mpv_equation_of_time_pvcdrom(
    dayofyear: Int, eot: Int, n: Int, day_angle: Int
) abi("C"):
    equation_of_time_pvcdrom(p(dayofyear), p(eot), n, p(day_angle))


@export("mpv_declination_spencer71")
def mpv_declination_spencer71(
    dayofyear: Int, dec: Int, n: Int, day_angle: Int
) abi("C"):
    declination_spencer71(p(dayofyear), p(dec), n, p(day_angle))


@export("mpv_declination_cooper69")
def mpv_declination_cooper69(
    dayofyear: Int, dec: Int, n: Int, day_angle: Int
) abi("C"):
    declination_cooper69(p(dayofyear), p(dec), n, p(day_angle))


@export("mpv_solar_azimuth_analytical")
def mpv_solar_azimuth_analytical(
    latitude: Int, hourangle: Int, declination: Int, zenith: Int, azimuth: Int,
    n: Int
) abi("C"):
    solar_azimuth_analytical(
        p(latitude), p(hourangle), p(declination), p(zenith), p(azimuth), n
    )


@export("mpv_solar_zenith_analytical")
def mpv_solar_zenith_analytical(
    latitude: Int, hourangle: Int, declination: Int, zenith: Int, n: Int
) abi("C"):
    solar_zenith_analytical(
        p(latitude), p(hourangle), p(declination), p(zenith), n
    )


@export("mpv_hour_angle")
def mpv_hour_angle(
    hrs_minus_tzs: Int, longitude: Float64, equation_of_time: Int,
    hourangle: Int, n: Int
) abi("C"):
    hour_angle(p(hrs_minus_tzs), longitude, p(equation_of_time), p(hourangle), n)


@export("mpv_sunset_angle")
def mpv_sunset_angle(latitude: Float64, declination: Int, angle: Int, n: Int) abi("C"):
    sunset_angle(latitude, p(declination), p(angle), n)


# ------------------------------------------------------------- atmosphere
@export("mpv_pres2alt")
def mpv_pres2alt(pressure: Int, alt: Int, n: Int) abi("C"):
    pres2alt(p(pressure), p(alt), n)


@export("mpv_alt2pres")
def mpv_alt2pres(altitude: Int, press: Int, n: Int) abi("C"):
    alt2pres(p(altitude), p(press), n)


@export("mpv_get_absolute_airmass")
def mpv_get_absolute_airmass(
    airmass_relative: Int, pressure: Int, airmass_absolute: Int, n: Int
) abi("C"):
    get_absolute_airmass(
        p(airmass_relative), p(pressure), p(airmass_absolute), n
    )


@export("mpv_get_relative_airmass")
def mpv_get_relative_airmass(zenith: Int, airmass: Int, n: Int, model: Int) abi("C"):
    get_relative_airmass(p(zenith), p(airmass), n, model)


@export("mpv_gueymard94_pw")
def mpv_gueymard94_pw(
    temp_air: Int, relative_humidity: Int, pw: Int, n: Int
) abi("C"):
    gueymard94_pw(p(temp_air), p(relative_humidity), p(pw), n)


@export("mpv_bird_hulstrom80_aod_bb")
def mpv_bird_hulstrom80_aod_bb(
    aod380: Int, aod500: Int, aod_bb: Int, n: Int
) abi("C"):
    bird_hulstrom80_aod_bb(p(aod380), p(aod500), p(aod_bb), n)


@export("mpv_kasten96_lt")
def mpv_kasten96_lt(
    airmass_absolute: Int, precipitable_water: Int, aod_bb: Int, lt: Int, n: Int
) abi("C"):
    kasten96_lt(p(airmass_absolute), p(precipitable_water), p(aod_bb), p(lt), n)



@export("mpv_rh_from_tdew")
def mpv_rh_from_tdew(
    temp_air: Int, temp_dew: Int, relative_humidity: Int, n: Int,
    coeff0: Float64, coeff1: Float64, coeff2: Float64,
) abi("C"):
    rh_from_tdew(
        p(temp_air), p(temp_dew), p(relative_humidity), n, coeff0, coeff1, coeff2
    )


@export("mpv_tdew_from_rh")
def mpv_tdew_from_rh(
    temp_air: Int, relative_humidity: Int, dewpoint: Int, n: Int,
    coeff0: Float64, coeff1: Float64, coeff2: Float64,
) abi("C"):
    tdew_from_rh(
        p(temp_air), p(relative_humidity), p(dewpoint), n, coeff0, coeff1, coeff2
    )


@export("mpv_angstrom_aod_at_lambda")
def mpv_angstrom_aod_at_lambda(
    aod0: Int, lambda0: Int, aod1: Int, n: Int, alpha: Float64, lambda1: Float64
) abi("C"):
    angstrom_aod_at_lambda(p(aod0), p(lambda0), p(aod1), n, alpha, lambda1)


@export("mpv_angstrom_alpha")
def mpv_angstrom_alpha(
    aod1: Int, lambda1: Int, aod2: Int, lambda2: Int, alpha_out: Int, n: Int
) abi("C"):
    angstrom_alpha(p(aod1), p(lambda1), p(aod2), p(lambda2), p(alpha_out), n)


@export("mpv_windspeed_powerlaw")
def mpv_windspeed_powerlaw(
    wind_speed_reference: Int, height_desired: Int,
    height_reference: Int, exponent: Float64, wind_speed: Int, n: Int,
) abi("C"):
    windspeed_powerlaw(
        p(wind_speed_reference), p(height_desired), p(height_reference), exponent,
        p(wind_speed), n,
    )

# ------------------------------------------------------------- irradiance
@export("mpv_get_extra_radiation")
def mpv_get_extra_radiation(
    doy: Int, dni_extra: Int, n: Int, solar_constant: Float64, method: Int
) abi("C"):
    get_extra_radiation(p(doy), p(dni_extra), n, solar_constant, method)


@export("mpv_aoi_projection")
def mpv_aoi_projection(
    surface_tilt: Int, surface_azimuth: Int, solar_zenith: Int, solar_azimuth: Int,
    projection: Int, n: Int
) abi("C"):
    aoi_projection(
        p(surface_tilt), p(surface_azimuth), p(solar_zenith), p(solar_azimuth),
        p(projection), n
    )


@export("mpv_aoi")
def mpv_aoi(
    surface_tilt: Int, surface_azimuth: Int, solar_zenith: Int, solar_azimuth: Int,
    dst: Int, n: Int
) abi("C"):
    aoi(p(surface_tilt), p(surface_azimuth), p(solar_zenith), p(solar_azimuth), p(dst), n)


@export("mpv_beam_component")
def mpv_beam_component(
    surface_tilt: Int, surface_azimuth: Int, solar_zenith: Int, solar_azimuth: Int,
    dni: Int, beam: Int, n: Int
) abi("C"):
    beam_component(
        p(surface_tilt), p(surface_azimuth), p(solar_zenith), p(solar_azimuth),
        p(dni), p(beam), n
    )


@export("mpv_get_ground_diffuse")
def mpv_get_ground_diffuse(
    surface_tilt: Int, ghi: Int, poa_ground_diffuse: Int, n: Int, albedo: Float64
) abi("C"):
    get_ground_diffuse(p(surface_tilt), p(ghi), p(poa_ground_diffuse), n, albedo)


@export("mpv_isotropic")
def mpv_isotropic(
    surface_tilt: Int, dhi: Int, poa_sky_diffuse: Int, poa_isotropic: Int, n: Int
) abi("C"):
    isotropic(p(surface_tilt), p(dhi), p(poa_sky_diffuse), p(poa_isotropic), n)


@export("mpv_klucher")
def mpv_klucher(
    surface_tilt: Int, surface_azimuth: Int, dhi: Int, ghi: Int, solar_zenith: Int,
    solar_azimuth: Int, poa_sky_diffuse: Int, n: Int
) abi("C"):
    klucher(
        p(surface_tilt), p(surface_azimuth), p(dhi), p(ghi), p(solar_zenith),
        p(solar_azimuth), p(poa_sky_diffuse), n
    )


@export("mpv_haydavies")
def mpv_haydavies(
    surface_tilt: Int, surface_azimuth: Int, dhi: Int, dni: Int, dni_extra: Int,
    solar_zenith: Int, solar_azimuth: Int, projection_ratio: Int,
    poa_sky_diffuse: Int, poa_isotropic: Int, poa_circumsolar: Int, n: Int,
    use_projection_ratio: Int
) abi("C"):
    haydavies(
        p(surface_tilt), p(surface_azimuth), p(dhi), p(dni), p(dni_extra),
        p(solar_zenith), p(solar_azimuth), p(projection_ratio), p(poa_sky_diffuse),
        p(poa_isotropic), p(poa_circumsolar), n, use_projection_ratio
    )


@export("mpv_reindl")
def mpv_reindl(
    surface_tilt: Int, surface_azimuth: Int, dhi: Int, dni: Int, ghi: Int,
    dni_extra: Int, solar_zenith: Int, solar_azimuth: Int, poa_sky_diffuse: Int,
    poa_isotropic: Int, poa_circumsolar: Int, poa_horizon: Int, n: Int
) abi("C"):
    reindl(
        p(surface_tilt), p(surface_azimuth), p(dhi), p(dni), p(ghi), p(dni_extra),
        p(solar_zenith), p(solar_azimuth), p(poa_sky_diffuse), p(poa_isotropic),
        p(poa_circumsolar), p(poa_horizon), n
    )


@export("mpv_king")
def mpv_king(
    surface_tilt: Int, dhi: Int, ghi: Int, solar_zenith: Int, poa_sky_diffuse: Int,
    n: Int
) abi("C"):
    king(p(surface_tilt), p(dhi), p(ghi), p(solar_zenith), p(poa_sky_diffuse), n)


@export("mpv_perez")
def mpv_perez(
    surface_tilt: Int, surface_azimuth: Int, dhi: Int, dni: Int, dni_extra: Int,
    solar_zenith: Int, solar_azimuth: Int, airmass: Int, f1c: Int, f2c: Int,
    poa_sky_diffuse: Int, poa_isotropic: Int, poa_circumsolar: Int, poa_horizon: Int,
    n: Int
) abi("C"):
    perez(
        p(surface_tilt), p(surface_azimuth), p(dhi), p(dni), p(dni_extra),
        p(solar_zenith), p(solar_azimuth), p(airmass), p(f1c), p(f2c),
        p(poa_sky_diffuse), p(poa_isotropic), p(poa_circumsolar), p(poa_horizon), n
    )


@export("mpv_clearsky_index")
def mpv_clearsky_index(
    ghi: Int, ghi_clear: Int, dst: Int, n: Int, max_clearsky_index: Float64
) abi("C"):
    clearsky_index(p(ghi), p(ghi_clear), p(dst), n, max_clearsky_index)


@export("mpv_clearness_index")
def mpv_clearness_index(
    ghi: Int, solar_zenith: Int, extra_radiation: Int, kt: Int, n: Int,
    min_cos_zenith: Float64, max_clearness_index: Float64
) abi("C"):
    clearness_index(
        p(ghi), p(solar_zenith), p(extra_radiation), p(kt), n, min_cos_zenith,
        max_clearness_index
    )


@export("mpv_clearness_index_zenith_independent")
def mpv_clearness_index_zenith_independent(
    clearness_index: Int, airmass: Int, kt_prime: Int, n: Int,
    max_clearness_index: Float64
) abi("C"):
    clearness_index_zenith_independent(
        p(clearness_index), p(airmass), p(kt_prime), n, max_clearness_index
    )


@export("mpv_disc")
def mpv_disc(
    ghi: Int, solar_zenith: Int, doy: Int, dni: Int, kt: Int, airmass: Int, n: Int,
    pressure: Float64, min_cos_zenith: Float64, max_zenith: Float64,
    max_airmass: Float64, use_pressure: Int
) abi("C"):
    disc(
        p(ghi), p(solar_zenith), p(doy), p(dni), p(kt), p(airmass), n, pressure,
        min_cos_zenith, max_zenith, max_airmass, use_pressure
    )


@export("mpv_dirint")
def mpv_dirint(
    ghi: Int, solar_zenith: Int, doy: Int, delta_kt_prime: Int, temp_dew: Int,
    coeffs: Int, dni: Int, n: Int, pressure: Float64, max_airmass: Float64,
    use_pressure: Int, use_delta_kt_prime: Int, use_temp_dew: Int,
    min_cos_zenith: Float64, max_zenith: Float64, scratch: Int
) abi("C"):
    dirint(
        p(ghi), p(solar_zenith), p(doy), p(delta_kt_prime), p(temp_dew), p(coeffs),
        p(dni), n, pressure, max_airmass, use_pressure, use_delta_kt_prime,
        use_temp_dew, min_cos_zenith, max_zenith, p(scratch)
    )


@export("mpv_dirindex")
def mpv_dirindex(
    dni_clear: Int, dni_dirint: Int, dni_dirint_clearsky: Int, dni_dirindex: Int,
    n: Int
) abi("C"):
    dirindex(p(dni_clear), p(dni_dirint), p(dni_dirint_clearsky), p(dni_dirindex), n)


@export("mpv_erbs")
def mpv_erbs(
    ghi: Int, zenith: Int, dni_extra: Int, dni: Int, dhi: Int, kt: Int, n: Int,
    min_cos_zenith: Float64, max_zenith: Float64
) abi("C"):
    erbs(
        p(ghi), p(zenith), p(dni_extra), p(dni), p(dhi), p(kt), n, min_cos_zenith,
        max_zenith
    )


@export("mpv_orgill_hollands")
def mpv_orgill_hollands(
    ghi: Int, zenith: Int, dni_extra: Int, dni: Int, dhi: Int, kt: Int, n: Int,
    min_cos_zenith: Float64, max_zenith: Float64
) abi("C"):
    orgill_hollands(
        p(ghi), p(zenith), p(dni_extra), p(dni), p(dhi), p(kt), n, min_cos_zenith,
        max_zenith
    )


@export("mpv_boland")
def mpv_boland(
    ghi: Int, solar_zenith: Int, dni_extra: Int, dni: Int, dhi: Int, kt: Int,
    n: Int, a_coeff: Float64, b_coeff: Float64, min_cos_zenith: Float64,
    max_zenith: Float64
) abi("C"):
    boland(
        p(ghi), p(solar_zenith), p(dni_extra), p(dni), p(dhi), p(kt), n, a_coeff,
        b_coeff, min_cos_zenith, max_zenith
    )


@export("mpv_dni")
def mpv_dni(
    ghi: Int, dhi: Int, zenith: Int, dni_clear: Int, dst: Int, n: Int,
    dni_clear_tolerance: Float64, use_dni_clear: Int,
    zenith_threshold_for_zero_dni: Float64, zenith_threshold_for_clearsky_limit: Float64
) abi("C"):
    dni(
        p(ghi), p(dhi), p(zenith), p(dni_clear), p(dst), n, dni_clear_tolerance,
        use_dni_clear, zenith_threshold_for_zero_dni, zenith_threshold_for_clearsky_limit
    )


@export("mpv_louche")
def mpv_louche(
    ghi: Int, solar_zenith: Int, dni_extra: Int, dni: Int, dhi: Int, kt: Int, n: Int,
    max_zenith: Float64
) abi("C"):
    louche(p(ghi), p(solar_zenith), p(dni_extra), p(dni), p(dhi), p(kt), n, max_zenith)


@export("mpv_campbell_norman")
def mpv_campbell_norman(
    zenith: Int,
    ghi: Int,
    dni: Int,
    dhi: Int,
    n: Int,
    transmittance: Float64,
    pressure: Float64,
    dni_extra: Float64,
) abi("C"):
    campbell_norman(
        p(zenith), p(ghi), p(dni), p(dhi), n, transmittance, pressure, dni_extra
    )


@export("mpv_diffuse_par_spitters")
def mpv_diffuse_par_spitters(
    daily_solar_zenith: Int,
    global_diffuse_fraction: Int,
    par_diffuse_fraction: Int,
    n: Int,
) abi("C"):
    diffuse_par_spitters(
        p(daily_solar_zenith), p(global_diffuse_fraction),
        p(par_diffuse_fraction), n,
    )


@export("mpv_poa_components")
def mpv_poa_components(
    aoi: Int,
    dni: Int,
    poa_sky_diffuse: Int,
    poa_ground_diffuse: Int,
    poa_direct: Int,
    poa_diffuse: Int,
    poa_global: Int,
    n: Int,
) abi("C"):
    poa_components(
        p(aoi), p(dni), p(poa_sky_diffuse), p(poa_ground_diffuse),
        p(poa_direct), p(poa_diffuse), p(poa_global), n,
    )


# ----------------------------------------------------------------- tables
@export("mpv_dirint_coeffs_table")
def mpv_dirint_coeffs_table(dst: Int) abi("C"):
    dirint_coeffs_table(p(dst))


@export("mpv_perez_coefficients")
def mpv_perez_coefficients(model: Int, f1: Int, f2: Int) abi("C"):
    perez_coefficients(model, p(f1), p(f2))
