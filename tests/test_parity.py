"""Parity against the real pvlib, on the same inputs, function by function.

Every test in this file compares mojo-pvlib against pvlib 0.16.x. Anything not
compared here is listed in the README's coverage section.
"""

import numpy as np
import pandas as pd
import pytest

pvlib_solarposition = pytest.importorskip("pvlib.solarposition")
import pvlib
import pvlib.atmosphere
import pvlib.irradiance
import pvlib.spa
import pvlib.tools

import mojopvlib as pv
from mojopvlib import atmosphere, irradiance, solarposition, spa, tools

# LLVM's pow agrees with libm's to ~1e-10 relative rather than exactly, and
# upstream's pres2alt subtracts two nearly equal numbers, so the models built on
# `**` need a looser relative tolerance than the rest.
POW_TOL = 1e-6
AIRMASS_TOL = 1e-8
DECOMP_TOL = 1e-9


def same(a, b, rtol=1e-12, atol=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


@pytest.fixture(scope="module")
def year():
    return pd.date_range("2020-01-01 00:00", periods=8760, freq="h", tz="UTC")


@pytest.fixture(scope="module")
def summer():
    return pd.date_range("2020-06-21 00:00", periods=240, freq="h", tz="UTC")


@pytest.fixture(scope="module")
def day():
    return pd.date_range("2020-06-21 00:00", periods=240, freq="h", tz="MST")


@pytest.fixture(scope="module")
def irr():
    rng = np.random.default_rng(0)
    n = 240
    solar_zenith = np.linspace(5, 85, n)
    ghi = np.clip(1000 * np.sin(np.radians(90 - solar_zenith)) + rng.normal(0, 5, n), 0, None)
    dhi = np.clip(0.25 * ghi + rng.normal(0, 3, n), 0, None)
    dni = np.clip((ghi - dhi) / np.cos(np.radians(solar_zenith)) + rng.normal(0, 5, n), 0, None)
    return dict(
        n=n,
        surface_tilt=np.full(n, 30.0),
        surface_azimuth=np.full(n, 180.0),
        solar_zenith=solar_zenith,
        solar_azimuth=np.linspace(0, 360, n),
        ghi=ghi,
        dhi=dhi,
        dni=dni,
        doy=np.full(n, 172.0),
    )


# ----------------------------------------------------------------- tools
def test_trig_helpers():
    angle = np.linspace(-720, 720, 1001)
    for ours, theirs in [
        (tools.cosd(angle), pvlib.tools.cosd(angle)),
        (tools.sind(angle), pvlib.tools.sind(angle)),
        (tools.tand(angle), pvlib.tools.tand(angle)),
    ]:
        same(ours, theirs, rtol=1e-12, atol=1e-12)
    number = np.linspace(-1, 1, 501)
    for ours, theirs in [
        (tools.asind(number), pvlib.tools.asind(number)),
        (tools.acosd(number), pvlib.tools.acosd(number)),
        (tools.atand(number), pvlib.tools.atand(number)),
    ]:
        same(ours, theirs, rtol=1e-12, atol=1e-12)


def test_trig_helpers_scalar():
    assert tools.cosd(180.0) == pytest.approx(pvlib.tools.cosd(180.0), abs=1e-15)
    assert tools.sind(-45.0) == pytest.approx(pvlib.tools.sind(-45.0), abs=1e-15)


# ---------------------------------------------------------- solarposition
def test_spa_python(summer):
    ours = solarposition.spa_python(summer, 35.0, -105.0, altitude=1600.0)
    theirs = pvlib.solarposition.spa_python(summer, 35.0, -105.0, altitude=1600.0)
    assert list(ours.columns) == list(theirs.columns)
    assert ours.index.equals(theirs.index)
    for col in theirs.columns:
        same(ours[col], theirs[col], rtol=0, atol=1e-8)


def test_spa_python_defaults_and_delta_t(summer):
    ours = solarposition.spa_python(summer, -33.9, 151.2, delta_t=None)
    theirs = pvlib.solarposition.spa_python(summer, -33.9, 151.2, delta_t=None)
    same(ours["zenith"], theirs["zenith"], rtol=0, atol=1e-8)
    same(ours["equation_of_time"], theirs["equation_of_time"], rtol=0, atol=1e-8)


def test_get_solarposition_nrel_numpy(summer):
    ours = solarposition.get_solarposition(summer, 64.0, -21.0, method="nrel_numpy")
    theirs = pvlib.solarposition.get_solarposition(
        summer, 64.0, -21.0, method="nrel_numpy")
    for col in ["zenith", "azimuth", "elevation", "apparent_zenith"]:
        same(ours[col], theirs[col], rtol=0, atol=1e-8)


def test_get_solarposition_derives_pressure(summer):
    ours = solarposition.get_solarposition(summer, 20.0, 30.0, altitude=1000.0)
    theirs = pvlib.solarposition.get_solarposition(
        summer, 20.0, 30.0, altitude=1000.0)
    same(ours["zenith"], theirs["zenith"], rtol=0, atol=1e-8)


def test_get_solarposition_rejects_unported_method(summer):
    with pytest.raises(ValueError):
        solarposition.get_solarposition(summer, 0.0, 0.0, method="pyephem")


def test_nrel_earthsun_distance(year):
    same(solarposition.nrel_earthsun_distance(year),
         pvlib.solarposition.nrel_earthsun_distance(year), rtol=1e-12)


def test_equation_of_time_spencer71():
    doy = np.arange(1, 367, dtype=float)
    same(solarposition.equation_of_time_spencer71(doy),
         pvlib.solarposition.equation_of_time_spencer71(doy), rtol=1e-12)


def test_equation_of_time_pvcdrom():
    doy = np.arange(1, 367, dtype=float)
    same(solarposition.equation_of_time_pvcdrom(doy),
         pvlib.solarposition.equation_of_time_pvcdrom(doy), rtol=1e-12)


def test_declination_spencer71():
    doy = np.arange(1, 367, dtype=float)
    same(solarposition.declination_spencer71(doy),
         pvlib.solarposition.declination_spencer71(doy), rtol=1e-12)


def test_declination_cooper69():
    doy = np.arange(1, 367, dtype=float)
    # declination crosses zero, so compare with an absolute floor
    same(solarposition.declination_cooper69(doy),
         pvlib.solarposition.declination_cooper69(doy), rtol=1e-10, atol=1e-12)


def test_calculate_simple_day_angle():
    doy = np.arange(1, 367, dtype=float)
    same(solarposition._calculate_simple_day_angle(doy),
         pvlib.solarposition._calculate_simple_day_angle(doy), rtol=1e-15)
    same(solarposition._calculate_simple_day_angle(doy, offset=0),
         pvlib.solarposition._calculate_simple_day_angle(doy, offset=0), rtol=1e-15)


def test_solar_azimuth_analytical():
    lat = np.radians(np.linspace(-89, 89, 97))
    ha = np.radians(np.linspace(-180, 180, 97))
    dec = np.radians(np.linspace(-23.4, 23.4, 97))
    same(solarposition.solar_azimuth_analytical(lat, ha, dec, ha),
         pvlib.solarposition.solar_azimuth_analytical(lat, ha, dec, ha),
         rtol=1e-12, atol=1e-12)


def test_solar_zenith_analytical():
    lat = np.radians(np.linspace(-89, 89, 97))
    ha = np.radians(np.linspace(-180, 180, 97))
    dec = np.radians(np.linspace(-23.4, 23.4, 97))
    same(solarposition.solar_zenith_analytical(lat, ha, dec),
         pvlib.solarposition.solar_zenith_analytical(lat, ha, dec),
         rtol=1e-12, atol=1e-12)


def test_hour_angle(day):
    eot = solarposition.equation_of_time_spencer71(day.dayofyear)
    same(solarposition.hour_angle(day, -105.0, eot),
         pvlib.solarposition.hour_angle(day, -105.0, eot), rtol=1e-12)


def test_hour_angle_requires_localized(year):
    with pytest.raises(ValueError):
        solarposition.hour_angle(year.tz_localize(None), 0.0, 0.0)


def test_sun_rise_set_transit_geometric(day):
    dec = solarposition.declination_cooper69(day.dayofyear)
    eot = solarposition.equation_of_time_spencer71(day.dayofyear)
    ours = solarposition.sun_rise_set_transit_geometric(day, 35.0, -105.0, dec, eot)
    theirs = pvlib.solarposition.sun_rise_set_transit_geometric(
        day, 35.0, -105.0, dec, eot)
    for a, b in zip(ours, theirs):
        assert np.max(np.abs((a - b).total_seconds())) < 1e-6


def test_calculate_deltat():
    year = np.array([-1000.0, 0.0, 500.0, 1000.0, 1700.0, 1900.0, 2020.0, 2100.0])
    month = np.full(8, 6.0)
    same(spa.calculate_deltat(year, month), pvlib.spa.calculate_deltat(year, month))
    assert spa.calculate_deltat(2020.0, 7.0) == pytest.approx(
        pvlib.spa.calculate_deltat(2020, 7))


def test_julian_day():
    unixtime = np.array([0.0, 86400.0, 1_600_000_000.0, 1_234_567_890.0])
    same(spa.julian_day(unixtime), pvlib.spa.julian_day(unixtime), rtol=1e-15)
    assert spa.julian_day(0.0) == pytest.approx(pvlib.spa.julian_day(0.0))


def test_spa_tables_match_upstream():
    for name in ["L0", "L1", "L2", "L3", "L4", "L5", "B0", "B1",
                 "R0", "R1", "R2", "R3", "R4",
                 "NUTATION_ABCD_ARRAY", "NUTATION_YTERM_ARRAY"]:
        same(pv._lib.spa_table(name), getattr(pvlib.spa, name), rtol=0, atol=0)


def test_dirint_coeffs_match_upstream():
    same(pv._lib.dirint_coeffs(), pvlib.irradiance._get_dirint_coeffs(), rtol=0, atol=0)


def test_perez_coefficients_match_upstream():
    from mojopvlib._lib import PEREZ_MODELS
    for model in PEREZ_MODELS:
        f1, f2 = pv._lib.perez_coefficients(model)
        g1, g2 = pvlib.irradiance._get_perez_coefficients(model)
        same(f1, g1, rtol=0, atol=0)
        same(f2, g2[:, :3], rtol=0, atol=0)


# ------------------------------------------------------------- atmosphere
@pytest.mark.parametrize("model", [
    "simple", "kasten1966", "kastenyoung1989", "youngirvine1967", "young1994",
    "gueymard1993", "gueymard2003", "pickering2002",
])
def test_get_relative_airmass(model):
    zenith = np.linspace(0, 95, 200)
    same(atmosphere.get_relative_airmass(zenith, model),
         pvlib.atmosphere.get_relative_airmass(zenith, model), rtol=AIRMASS_TOL)


def test_get_relative_airmass_invalid_model():
    with pytest.raises(ValueError):
        atmosphere.get_relative_airmass(np.array([10.0]), "nope")


def test_get_absolute_airmass():
    am = np.linspace(1, 5, 50)
    pressure = np.linspace(70000.0, 101325.0, 50)
    same(atmosphere.get_absolute_airmass(am, pressure),
         pvlib.atmosphere.get_absolute_airmass(am, pressure), rtol=1e-14)


def test_gueymard94_pw():
    t = np.linspace(-10.0, 45.0, 50)
    rh = np.linspace(5.0, 100.0, 50)
    same(atmosphere.gueymard94_pw(t, rh), pvlib.atmosphere.gueymard94_pw(t, rh),
         rtol=1e-11)


def test_bird_hulstrom80_aod_bb():
    a380 = np.linspace(0.01, 0.5, 20)
    a500 = np.linspace(0.01, 0.3, 20)
    same(atmosphere.bird_hulstrom80_aod_bb(a380, a500),
         pvlib.atmosphere.bird_hulstrom80_aod_bb(a380, a500), rtol=1e-15)


def test_kasten96_lt():
    am = np.linspace(1.0, 5.0, 20)
    pw = np.linspace(0.5, 5.0, 20)
    aod = np.linspace(0.01, 0.3, 20)
    same(atmosphere.kasten96_lt(am, pw, aod), pvlib.atmosphere.kasten96_lt(am, pw, aod),
         rtol=POW_TOL)


def test_pres2alt():
    pressure = np.linspace(70000.0, 105000.0, 50)
    same(atmosphere.pres2alt(pressure), pvlib.atmosphere.pres2alt(pressure),
         rtol=POW_TOL)


def test_alt2pres():
    altitude = np.linspace(0.0, 4000.0, 50)
    same(atmosphere.alt2pres(altitude), pvlib.atmosphere.alt2pres(altitude),
         rtol=POW_TOL)


# ------------------------------------------------------------- irradiance
def test_get_extra_radiation_spencer():
    doy = np.arange(1, 366, dtype=float)
    same(irradiance.get_extra_radiation(doy), pvlib.irradiance.get_extra_radiation(doy),
         rtol=1e-12)
    assert irradiance.get_extra_radiation(172.0) == pytest.approx(
        pvlib.irradiance.get_extra_radiation(172.0), rel=1e-12)


def test_get_extra_radiation_asce():
    doy = np.arange(1, 366, dtype=float)
    same(irradiance.get_extra_radiation(doy, method="asce"),
         pvlib.irradiance.get_extra_radiation(doy, method="asce"), rtol=1e-12)


def test_get_extra_radiation_nrel(year):
    ours = irradiance.get_extra_radiation(year, method="nrel")
    theirs = pvlib.irradiance.get_extra_radiation(year, method="nrel")
    assert ours.index.equals(theirs.index)
    same(ours, theirs, rtol=1e-12)


def test_get_extra_radiation_invalid_method():
    with pytest.raises(ValueError):
        irradiance.get_extra_radiation(np.array([1.0]), method="nope")


def test_aoi_projection(irr):
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["solar_zenith"],
         irr["solar_azimuth"])
    same(irradiance.aoi_projection(*f), pvlib.irradiance.aoi_projection(*f), rtol=1e-13)


def test_aoi(irr):
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["solar_zenith"],
         irr["solar_azimuth"])
    same(irradiance.aoi(*f), pvlib.irradiance.aoi(*f), rtol=1e-12)


def test_aoi_projection_scalars():
    assert irradiance.aoi_projection(30.0, 180.0, 45.0, 180.0) == pytest.approx(
        pvlib.irradiance.aoi_projection(30.0, 180.0, 45.0, 180.0), abs=1e-14)


def test_beam_component(irr):
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["solar_zenith"],
         irr["solar_azimuth"], irr["dni"])
    same(irradiance.beam_component(*f), pvlib.irradiance.beam_component(*f),
         rtol=1e-13)


def test_get_ground_diffuse(irr):
    same(irradiance.get_ground_diffuse(irr["surface_tilt"], irr["ghi"]),
         pvlib.irradiance.get_ground_diffuse(irr["surface_tilt"], irr["ghi"]),
         rtol=0, atol=0)
    same(irradiance.get_ground_diffuse(20.0, irr["ghi"], albedo=0.4),
         pvlib.irradiance.get_ground_diffuse(20.0, irr["ghi"], albedo=0.4), rtol=0, atol=0)


def test_get_ground_diffuse_surface_type(irr):
    # The albedo table is inlined, so it is checked against the whole of
    # upstream's, key for key, not just one entry.
    import pvlib.albedo
    assert irradiance._SURFACE_ALBEDOS == pvlib.albedo.SURFACE_ALBEDOS
    for surface_type in pvlib.albedo.SURFACE_ALBEDOS:
        ours = irradiance.get_ground_diffuse(20.0, irr["ghi"],
                                             surface_type=surface_type)
        theirs = pvlib.irradiance.get_ground_diffuse(20.0, irr["ghi"],
                                                     surface_type=surface_type)
        same(ours, theirs, rtol=0, atol=0)


def test_isotropic(irr):
    ours = irradiance.isotropic(irr["surface_tilt"], irr["dhi"])
    theirs = pvlib.irradiance.isotropic(irr["surface_tilt"], irr["dhi"])
    same(ours, theirs, rtol=0, atol=0)


def test_isotropic_components(irr):
    ours = irradiance.isotropic(30.0, irr["dhi"], return_components=True)
    theirs = pvlib.irradiance.isotropic(30.0, irr["dhi"], return_components=True)
    for key in theirs:
        same(ours[key], theirs[key], rtol=0, atol=0)


def test_klucher(irr):
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["dhi"], irr["ghi"],
         irr["solar_zenith"], irr["solar_azimuth"])
    same(irradiance.klucher(*f), pvlib.irradiance.klucher(*f), rtol=1e-12)


def test_haydavies(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["dhi"], irr["dni"], ex,
         irr["solar_zenith"], irr["solar_azimuth"])
    same(irradiance.haydavies(*f), pvlib.irradiance.haydavies(*f), rtol=1e-12)


def test_haydavies_components(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["dhi"], irr["dni"], ex,
         irr["solar_zenith"], irr["solar_azimuth"])
    ours = irradiance.haydavies(*f, return_components=True)
    theirs = pvlib.irradiance.haydavies(*f, return_components=True)
    for key in theirs:
        same(ours[key], theirs[key], rtol=1e-12)


def test_haydavies_projection_ratio(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    ratio = np.full(irr["n"], 1.2)
    ours = irradiance.haydavies(30.0, 180.0, irr["dhi"], irr["dni"], ex,
                                irr["solar_zenith"], irr["solar_azimuth"],
                                projection_ratio=ratio)
    theirs = pvlib.irradiance.haydavies(30.0, 180.0, irr["dhi"], irr["dni"], ex,
                                        irr["solar_zenith"], irr["solar_azimuth"],
                                        projection_ratio=ratio)
    same(ours, theirs, rtol=1e-12)


def test_reindl(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["dhi"], irr["dni"],
         irr["ghi"], ex, irr["solar_zenith"], irr["solar_azimuth"])
    same(irradiance.reindl(*f), pvlib.irradiance.reindl(*f), rtol=1e-12)


def test_reindl_components(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["dhi"], irr["dni"],
         irr["ghi"], ex, irr["solar_zenith"], irr["solar_azimuth"])
    ours = irradiance.reindl(*f, return_components=True)
    theirs = pvlib.irradiance.reindl(*f, return_components=True)
    for key in theirs:
        same(ours[key], theirs[key], rtol=1e-12)


def test_king(irr):
    with pytest.warns(pvlib._deprecation.pvlibDeprecationWarning):
        theirs = pvlib.irradiance.king(irr["surface_tilt"], irr["dhi"], irr["ghi"],
                                      irr["solar_zenith"])
    same(irradiance.king(irr["surface_tilt"], irr["dhi"], irr["ghi"],
                        irr["solar_zenith"]), theirs, rtol=1e-15, atol=0)


def test_perez(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    am = atmosphere.get_relative_airmass(irr["solar_zenith"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["dhi"], irr["dni"], ex,
         irr["solar_zenith"], irr["solar_azimuth"], am)
    same(irradiance.perez(*f), pvlib.irradiance.perez(*f), rtol=1e-12)


@pytest.mark.parametrize("model", [
    "allsitescomposite1990", "allsitescomposite1988", "sandiacomposite1988",
    "usacomposite1988", "france1988", "phoenix1988", "elmonte1988", "osage1988",
    "albuquerque1988", "albany1988",
])
def test_perez_models(irr, model):
    ex = irradiance.get_extra_radiation(irr["doy"])
    am = atmosphere.get_relative_airmass(irr["solar_zenith"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["dhi"], irr["dni"], ex,
         irr["solar_zenith"], irr["solar_azimuth"], am)
    same(irradiance.perez(*f, model=model), pvlib.irradiance.perez(*f, model=model),
         rtol=1e-12)


def test_perez_components(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    am = atmosphere.get_relative_airmass(irr["solar_zenith"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["dhi"], irr["dni"], ex,
         irr["solar_zenith"], irr["solar_azimuth"], am)
    ours = irradiance.perez(*f, return_components=True)
    theirs = pvlib.irradiance.perez(*f, return_components=True)
    for key in theirs:
        same(ours[key], theirs[key], rtol=1e-12)


def test_perez_nan_eps_matches(irr):
    # dhi = 0 makes eps NaN upstream, which must stay NaN here too
    dhi = np.zeros(irr["n"])
    ex = irradiance.get_extra_radiation(irr["doy"])
    am = atmosphere.get_relative_airmass(irr["solar_zenith"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], dhi, irr["dni"], ex,
         irr["solar_zenith"], irr["solar_azimuth"], am)
    same(irradiance.perez(*f), pvlib.irradiance.perez(*f))


def test_clearsky_index(irr):
    ghi_clear = irr["ghi"] * 0.95 + 10.0
    ours = irradiance.clearsky_index(irr["ghi"], ghi_clear)
    theirs = pvlib.irradiance.clearsky_index(irr["ghi"], ghi_clear)
    same(ours, theirs, rtol=0, atol=0)


def test_clearsky_index_preserves_nan(irr):
    ghi = irr["ghi"].copy()
    ghi[3] = np.nan
    ours = np.asarray(irradiance.clearsky_index(ghi, irr["ghi"]))
    theirs = np.asarray(pvlib.irradiance.clearsky_index(ghi, irr["ghi"]))
    same(ours, theirs)


def test_clearness_index(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    ours = irradiance.clearness_index(irr["ghi"], irr["solar_zenith"], ex)
    theirs = pvlib.irradiance.clearness_index(irr["ghi"], irr["solar_zenith"], ex)
    same(ours, theirs, rtol=0, atol=0)


def test_clearness_index_zenith_independent(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    am = atmosphere.get_relative_airmass(irr["solar_zenith"])
    kt = irradiance.clearness_index(irr["ghi"], irr["solar_zenith"], ex)
    same(irradiance.clearness_index_zenith_independent(kt, am),
         pvlib.irradiance.clearness_index_zenith_independent(kt, am),
         rtol=1e-12)


def test_disc(irr):
    ours = irradiance.disc(irr["ghi"], irr["solar_zenith"], irr["doy"])
    theirs = pvlib.irradiance.disc(irr["ghi"], irr["solar_zenith"], irr["doy"])
    for key in theirs:
        same(ours[key], theirs[key], rtol=DECOMP_TOL)


def test_disc_datetime_index(summer):
    ours = irradiance.disc(np.full(240, 500.0), np.full(240, 45.0), summer)
    theirs = pvlib.irradiance.disc(np.full(240, 500.0), np.full(240, 45.0), summer)
    assert isinstance(ours, pd.DataFrame)
    assert ours.index.equals(theirs.index)
    same(ours["dni"], theirs["dni"], rtol=DECOMP_TOL)


def test_disc_no_pressure(irr):
    ours = irradiance.disc(irr["ghi"], irr["solar_zenith"], irr["doy"], pressure=None)
    theirs = pvlib.irradiance.disc(irr["ghi"], irr["solar_zenith"], irr["doy"],
                                  pressure=None)
    for key in theirs:
        same(ours[key], theirs[key], rtol=DECOMP_TOL)


@pytest.mark.parametrize("name", ["erbs", "orgill_hollands", "boland", "louche"])
def test_decompositions(irr, name):
    ours = getattr(irradiance, name)(irr["ghi"], irr["solar_zenith"], irr["doy"])
    theirs = getattr(pvlib.irradiance, name)(irr["ghi"], irr["solar_zenith"],
                                             irr["doy"])
    for key in theirs:
        same(ours[key], theirs[key], rtol=DECOMP_TOL)


@pytest.mark.parametrize("name", ["erbs", "orgill_hollands", "boland", "louche"])
def test_decompositions_datetime_index(summer, name):
    ghi = np.full(240, 500.0)
    zenith = np.full(240, 45.0)
    ours = getattr(irradiance, name)(ghi, zenith, summer)
    theirs = getattr(pvlib.irradiance, name)(ghi, zenith, summer)
    assert isinstance(ours, pd.DataFrame)
    same(ours["dni"], theirs["dni"], rtol=DECOMP_TOL)
    same(ours["dhi"], theirs["dhi"], rtol=DECOMP_TOL)


def test_boland_coefficients(irr):
    ours = irradiance.boland(irr["ghi"], irr["solar_zenith"], irr["doy"],
                             a_coeff=7.997, b_coeff=0.586)
    theirs = pvlib.irradiance.boland(irr["ghi"], irr["solar_zenith"], irr["doy"],
                                     a_coeff=7.997, b_coeff=0.586)
    same(ours["dni"], theirs["dni"], rtol=1e-11)


def test_orgill_hollands_dni_extra(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    same(irradiance.orgill_hollands(irr["ghi"], irr["solar_zenith"], irr["doy"],
                                    dni_extra=ex)["dni"],
         pvlib.irradiance.orgill_hollands(irr["ghi"], irr["solar_zenith"],
                                          irr["doy"], dni_extra=ex)["dni"],
         rtol=1e-13)


def test_dirint(irr, summer):
    ours = irradiance.dirint(irr["ghi"], irr["solar_zenith"], summer)
    theirs = pvlib.irradiance.dirint(irr["ghi"], irr["solar_zenith"], summer)
    same(ours, theirs, rtol=DECOMP_TOL)


def test_dirint_temp_dew(irr, summer):
    temp_dew = np.linspace(-5.0, 25.0, irr["n"])
    ours = irradiance.dirint(irr["ghi"], irr["solar_zenith"], summer,
                             temp_dew=temp_dew)
    theirs = pvlib.irradiance.dirint(irr["ghi"], irr["solar_zenith"], summer,
                                     temp_dew=temp_dew)
    same(ours, theirs, rtol=DECOMP_TOL)


def test_dirint_no_delta_kt_prime(irr, summer):
    ours = irradiance.dirint(irr["ghi"], irr["solar_zenith"], summer,
                             use_delta_kt_prime=False)
    theirs = pvlib.irradiance.dirint(irr["ghi"], irr["solar_zenith"], summer,
                                     use_delta_kt_prime=False)
    same(ours, theirs, rtol=DECOMP_TOL)


def test_dirint_boundaries():
    # a three-point series exercises the boundary/gap branch of delta_kt_prime
    times = pd.date_range("2020-06-21", periods=3, freq="h", tz="UTC")
    ghi = np.array([100.0, 600.0, 200.0])
    zenith = np.array([60.0, 30.0, 70.0])
    same(irradiance.dirint(ghi, zenith, times),
         pvlib.irradiance.dirint(ghi, zenith, times), rtol=DECOMP_TOL)


def test_dirindex(irr, summer):
    ghi_clear = irr["ghi"] * 1.02 + 20.0
    ours = irradiance.dirindex(irr["ghi"], ghi_clear, irr["dni"], irr["solar_zenith"],
                               summer)
    theirs = pvlib.irradiance.dirindex(irr["ghi"], ghi_clear, irr["dni"],
                                       irr["solar_zenith"], summer)
    same(ours, theirs, rtol=1e-9)


def test_dni(irr):
    ours = irradiance.dni(irr["ghi"], irr["dhi"], irr["solar_zenith"])
    theirs = pvlib.irradiance.dni(irr["ghi"], irr["dhi"], irr["solar_zenith"])
    same(ours, theirs, rtol=0, atol=0)


def test_dni_clearsky_limit(irr):
    # upstream's array dni_clear path is broken (boolean assignment), so the
    # scalar form is what there is to compare against
    ours = irradiance.dni(irr["ghi"], irr["dhi"], irr["solar_zenith"],
                          dni_clear=900.0)
    theirs = pvlib.irradiance.dni(irr["ghi"], irr["dhi"], irr["solar_zenith"],
                                  dni_clear=900.0)
    same(ours, theirs, rtol=0, atol=0)


def test_complete_irradiance(irr):
    ours = irradiance.complete_irradiance(irr["solar_zenith"], ghi=irr["ghi"],
                                          dhi=irr["dhi"])
    theirs = pvlib.irradiance.complete_irradiance(irr["solar_zenith"], ghi=irr["ghi"],
                                                  dhi=irr["dhi"])
    for key in theirs:
        same(ours[key], theirs[key], rtol=0, atol=0)


def test_complete_irradiance_requires_exactly_one_missing(irr):
    with pytest.raises(ValueError):
        irradiance.complete_irradiance(irr["solar_zenith"], ghi=irr["ghi"])


@pytest.mark.parametrize("model", [
    "isotropic", "klucher", "haydavies", "reindl", "king", "perez",
])
def test_get_sky_diffuse(irr, model):
    ex = irradiance.get_extra_radiation(irr["doy"])
    am = atmosphere.get_relative_airmass(irr["solar_zenith"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["solar_zenith"],
         irr["solar_azimuth"], irr["dni"], irr["ghi"], irr["dhi"])
    kw = dict(dni_extra=ex, airmass=am, model=model)
    ours = irradiance.get_sky_diffuse(*f, **kw)
    if model == "king":
        with pytest.warns(pvlib._deprecation.pvlibDeprecationWarning):
            theirs = pvlib.irradiance.get_sky_diffuse(*f, **kw)
    else:
        theirs = pvlib.irradiance.get_sky_diffuse(*f, **kw)
    same(ours, theirs, rtol=1e-12)


def test_get_sky_diffuse_requires_dni_extra(irr):
    with pytest.raises(ValueError):
        irradiance.get_sky_diffuse(30.0, 180.0, 45.0, 180.0, 500.0, 800.0, 200.0,
                                   model="haydavies")


def test_get_sky_diffuse_rejects_components(irr):
    with pytest.raises(ValueError):
        irradiance.get_sky_diffuse(30.0, 180.0, 45.0, 180.0, 500.0, 800.0, 200.0,
                                   model="klucher", return_components=True)


def test_get_total_irradiance(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    am = atmosphere.get_relative_airmass(irr["solar_zenith"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["solar_zenith"],
         irr["solar_azimuth"], irr["dni"], irr["ghi"], irr["dhi"])
    ours = irradiance.get_total_irradiance(*f, dni_extra=ex, airmass=am,
                                           model="haydavies")
    theirs = pvlib.irradiance.get_total_irradiance(*f, dni_extra=ex, airmass=am,
                                                   model="haydavies")
    for key in theirs:
        same(ours[key], theirs[key], rtol=1e-12)


def test_get_total_irradiance_components(irr):
    ex = irradiance.get_extra_radiation(irr["doy"])
    am = atmosphere.get_relative_airmass(irr["solar_zenith"])
    f = (irr["surface_tilt"], irr["surface_azimuth"], irr["solar_zenith"],
         irr["solar_azimuth"], irr["dni"], irr["ghi"], irr["dhi"])
    ours = irradiance.get_total_irradiance(*f, dni_extra=ex, airmass=am,
                                           model="reindl", diffuse_components=True)
    theirs = pvlib.irradiance.get_total_irradiance(
        *f, dni_extra=ex, airmass=am, model="reindl", diffuse_components=True)
    for key in theirs:
        same(ours[key], theirs[key], rtol=1e-12)


# ------------------------------------------------------------ return types
def test_series_in_series_out(day):
    sp = solarposition.spa_python(day, 35.0, -105.0)
    ghi = pd.Series(np.full(240, 500.0), index=day)
    kt = irradiance.clearness_index(ghi, 45.0, 1367.0)
    assert isinstance(kt, pd.Series)
    assert kt.index.equals(day)


def test_spa_python_dataframe_columns(summer):
    sp = solarposition.spa_python(summer, 0.0, 0.0)
    assert list(sp.columns) == [
        "apparent_zenith", "zenith", "apparent_elevation", "elevation",
        "azimuth", "equation_of_time",
    ]
    assert sp.index.equals(summer)


# ------------------------------------------------- scalar/array broadcasting
# pvlib relies on numpy broadcasting, so every one of these has to accept a
# scalar where an array is expected and produce the full-length result.

def test_aoi_projection_mixed_scalars(irr):
    ours = irradiance.aoi_projection(30.0, 180.0, irr["solar_zenith"],
                                     irr["solar_azimuth"])
    theirs = pvlib.irradiance.aoi_projection(30.0, 180.0, irr["solar_zenith"],
                                             irr["solar_azimuth"])
    same(ours, theirs, rtol=1e-13)


def test_haydavies_mixed_scalars(irr):
    ex = float(irradiance.get_extra_radiation(172.0))
    ours = irradiance.haydavies(30.0, 180.0, irr["dhi"], irr["dni"], ex,
                                irr["solar_zenith"], irr["solar_azimuth"])
    theirs = pvlib.irradiance.haydavies(30.0, 180.0, irr["dhi"], irr["dni"], ex,
                                        irr["solar_zenith"], irr["solar_azimuth"])
    same(ours, theirs, rtol=1e-12)


def test_reindl_mixed_scalars(irr):
    ex = float(irradiance.get_extra_radiation(172.0))
    ours = irradiance.reindl(irr["surface_tilt"], 180.0, irr["dhi"], irr["dni"],
                             irr["ghi"], ex, irr["solar_zenith"],
                             irr["solar_azimuth"])
    theirs = pvlib.irradiance.reindl(irr["surface_tilt"], 180.0, irr["dhi"],
                                     irr["dni"], irr["ghi"], ex, irr["solar_zenith"],
                                     irr["solar_azimuth"])
    same(ours, theirs, rtol=1e-12)


def test_perez_mixed_scalars(irr):
    ex = float(irradiance.get_extra_radiation(172.0))
    am = float(atmosphere.get_relative_airmass(45.0))
    ours = irradiance.perez(30.0, 180.0, irr["dhi"], irr["dni"], ex,
                            irr["solar_zenith"], irr["solar_azimuth"], am)
    theirs = pvlib.irradiance.perez(30.0, 180.0, irr["dhi"], irr["dni"], ex,
                                    irr["solar_zenith"], irr["solar_azimuth"], am)
    same(ours, theirs, rtol=1e-12)


def test_dirint_mixed_scalars(irr, summer):
    ours = irradiance.dirint(irr["ghi"], irr["solar_zenith"], summer, temp_dew=10.0)
    theirs = pvlib.irradiance.dirint(irr["ghi"], irr["solar_zenith"], summer,
                                     temp_dew=10.0)
    same(ours, theirs, rtol=DECOMP_TOL)


def test_disc_scalar_datetime_or_doy(irr):
    ours = irradiance.disc(irr["ghi"], irr["solar_zenith"], 172.0)
    theirs = pvlib.irradiance.disc(irr["ghi"], irr["solar_zenith"], 172.0)
    for key in theirs:
        same(ours[key], theirs[key], rtol=DECOMP_TOL)


def test_decomposition_scalar_ghi():
    ours = irradiance.erbs(500.0, 45.0, 172.0)
    theirs = pvlib.irradiance.erbs(500.0, 45.0, 172.0)
    for key in theirs:
        assert ours[key] == pytest.approx(theirs[key], rel=1e-12)


def test_get_ground_diffuse_scalar_ghi():
    assert irradiance.get_ground_diffuse(30.0, 800.0) == pytest.approx(
        pvlib.irradiance.get_ground_diffuse(30.0, 800.0), abs=1e-12)


def test_haydavies_projection_ratio_overrides_angles(irr):
    ratio = np.linspace(0.5, 1.5, irr["n"])
    ours = irradiance.haydavies(30.0, 180.0, irr["dhi"], irr["dni"], 1366.0,
                                projection_ratio=ratio)
    theirs = pvlib.irradiance.haydavies(30.0, 180.0, irr["dhi"], irr["dni"], 1366.0,
                                        projection_ratio=ratio)
    same(ours, theirs, rtol=1e-12)


def test_get_total_irradiance_broadcasts_scalars_against_series(summer):
    sp = solarposition.spa_python(summer, 35.0, -105.0)
    ex = irradiance.get_extra_radiation(summer)
    ours = irradiance.get_total_irradiance(
        25, 180, sp["zenith"], sp["azimuth"], 800.0, 900.0, 100.0, dni_extra=ex,
        model="haydavies")
    theirs = pvlib.irradiance.get_total_irradiance(
        25, 180, sp["zenith"], sp["azimuth"], 800.0, 900.0, 100.0, dni_extra=ex,
        model="haydavies")
    # upstream wraps the totals in a DataFrame because `aoi` is a Series, even
    # though `dni` is scalar; the port has to make the same call.
    assert isinstance(ours, pd.DataFrame) and isinstance(theirs, pd.DataFrame)
    assert list(ours.columns) == list(theirs.columns)
    for key in theirs:
        same(ours[key], theirs[key], rtol=1e-12)


def test_aoi_projection_series_in_series_out(summer):
    sp = solarposition.spa_python(summer, 35.0, -105.0)
    ours = irradiance.aoi_projection(25.0, 180.0, sp["zenith"], sp["azimuth"])
    theirs = pvlib.irradiance.aoi_projection(25.0, 180.0, sp["zenith"],
                                             sp["azimuth"])
    assert isinstance(ours, pd.Series)
    same(ours, theirs, rtol=1e-13)


# ------------------------------------------------------------ parallel path
#
# Every kernel hands its element loop to `max.algorithm.parallelize` once the
# series is long enough and runs the same loop serially below that (see the
# `PAR_MIN_*` thresholds in `src/types.mojo`). Two things therefore need
# covering that the short-input tests above do not: that the parallel result
# is bit-identical to the serial one, and that both still match pvlib at those
# sizes. The kernels are elementwise, so tiling a short period across a long
# series gives exactly that: the long call is above every threshold and the
# short one is below all of them.

PAR_MIN_TINY = 1024
# A length above the largest threshold, so even the cheapest kernels dispatch.
LONG = 1 << 21


def tile(period, n=LONG):
    """`period` repeated to at least `n` elements."""
    return np.tile(period, int(np.ceil(n / np.size(period))))


def _columns(out):
    if isinstance(out, pd.DataFrame):
        return {c: np.asarray(out[c], dtype=float) for c in out.columns}
    if isinstance(out, dict):
        return {k: np.asarray(v, dtype=float) for k, v in out.items()}
    return {"": np.asarray(out, dtype=float)}


def parity(ours, theirs, period, rtol):
    """Check `ours` over a long tiled series against pvlib and against itself.

    `ours(period)` is the serial path and `ours(tile(period))` the parallel
    one; they must agree, and so must both agree with `theirs`.
    """
    reps = int(np.ceil(LONG / period.size))
    short = _columns(ours(period))
    long_ = _columns(ours(tile(period)))
    ref = _columns(theirs(tile(period)))
    first = next(iter(long_.values()))
    assert first.size >= LONG
    for key, expected in ref.items():
        same(long_[key][:LONG], expected[:LONG], rtol=rtol)
        same(long_[key][:LONG], np.tile(short[key], reps)[:LONG], rtol=rtol)


@pytest.fixture(scope="module")
def big_irr():
    n = 32
    solar_zenith = np.linspace(5, 85, n)
    ghi = np.clip(1000 * np.sin(np.radians(90 - solar_zenith)), 0, None)
    return dict(
        surface_tilt=np.full(n, 30.0),
        surface_azimuth=np.full(n, 180.0),
        solar_zenith=solar_zenith,
        solar_azimuth=np.linspace(0, 360, n),
        ghi=ghi,
        dhi=0.25 * ghi,
        dni=ghi * np.cos(np.radians(solar_zenith)),
        doy=np.full(n, 172.0),
    )


def test_parallel_tools_cosd():
    period = np.linspace(-720.0, 720.0, 37)
    parity(tools.cosd, pvlib.tools.cosd, period, rtol=1e-13)
    parity(tools.sind, pvlib.tools.sind, period, rtol=1e-13)
    parity(tools.tand, pvlib.tools.tand, period, rtol=1e-13)


def test_parallel_get_extra_radiation(big_irr):
    parity(irradiance.get_extra_radiation,
           pvlib.irradiance.get_extra_radiation, big_irr["doy"], rtol=1e-13)


def test_parallel_get_relative_airmass(big_irr):
    parity(atmosphere.get_relative_airmass,
           pvlib.atmosphere.get_relative_airmass, big_irr["solar_zenith"],
           rtol=AIRMASS_TOL)


def test_parallel_aoi_projection(big_irr):
    d = big_irr

    def args(z):
        return (np.resize(d["surface_tilt"], z.size),
                np.resize(d["surface_azimuth"], z.size), z,
                np.resize(d["solar_azimuth"], z.size))

    parity(lambda z: irradiance.aoi_projection(*args(z)),
           lambda z: pvlib.irradiance.aoi_projection(*args(z)),
           d["solar_zenith"], rtol=1e-12)


def test_parallel_isotropic(big_irr):
    parity(lambda h: irradiance.isotropic(
               np.resize(big_irr["surface_tilt"], h.size), h),
           lambda h: pvlib.irradiance.isotropic(30.0, h),
           big_irr["dhi"], rtol=1e-12)


def test_parallel_dni_from_ghi_dhi(big_irr):
    d = big_irr

    def args(z):
        return (np.resize(d["ghi"], z.size), np.resize(d["dhi"], z.size), z)

    parity(lambda z: irradiance.dni(*args(z)),
           lambda z: pvlib.irradiance.dni(*args(z)),
           d["solar_zenith"], rtol=1e-12)


def test_parallel_disc(big_irr):
    d = big_irr

    def args(z):
        return (np.resize(d["ghi"], z.size), z, np.resize(d["doy"], z.size))

    parity(lambda z: irradiance.disc(*args(z)),
           lambda z: pvlib.irradiance.disc(*args(z)),
           d["solar_zenith"], rtol=DECOMP_TOL)


def test_parallel_erbs(big_irr):
    d = big_irr

    def args(z):
        return (np.resize(d["ghi"], z.size), z, np.resize(d["doy"], z.size))

    parity(lambda z: irradiance.erbs(*args(z)),
           lambda z: pvlib.irradiance.erbs(*args(z)),
           d["solar_zenith"], rtol=DECOMP_TOL)


def test_parallel_boland(big_irr):
    d = big_irr

    def args(z):
        return (np.resize(d["ghi"], z.size), z, np.resize(d["doy"], z.size))

    parity(lambda z: irradiance.boland(*args(z)),
           lambda z: pvlib.irradiance.boland(*args(z)),
           d["solar_zenith"], rtol=DECOMP_TOL)


def test_parallel_haydavies(big_irr):
    d = big_irr
    extra = pvlib.irradiance.get_extra_radiation(d["doy"])

    def args(z):
        return (np.resize(d["surface_tilt"], z.size),
                np.resize(d["surface_azimuth"], z.size),
                np.resize(d["dhi"], z.size), np.resize(d["dni"], z.size),
                np.resize(extra, z.size), z,
                np.resize(d["solar_azimuth"], z.size))

    parity(lambda z: irradiance.haydavies(*args(z), return_components=True),
           lambda z: pvlib.irradiance.haydavies(*args(z), return_components=True),
           d["solar_zenith"], rtol=1e-12)


def test_parallel_reindl(big_irr):
    d = big_irr
    extra = pvlib.irradiance.get_extra_radiation(d["doy"])

    def args(z):
        return (np.resize(d["surface_tilt"], z.size),
                np.resize(d["surface_azimuth"], z.size),
                np.resize(d["dhi"], z.size), np.resize(d["dni"], z.size),
                np.resize(d["ghi"], z.size), np.resize(extra, z.size), z,
                np.resize(d["solar_azimuth"], z.size))

    parity(lambda z: irradiance.reindl(*args(z)),
           lambda z: pvlib.irradiance.reindl(*args(z)),
           d["solar_zenith"], rtol=1e-12)


def test_parallel_perez(big_irr):
    d = big_irr
    extra = pvlib.irradiance.get_extra_radiation(d["doy"])
    airmass = pv.atmosphere.get_relative_airmass(d["solar_zenith"])

    def args(z):
        return (np.resize(d["surface_tilt"], z.size),
                np.resize(d["surface_azimuth"], z.size),
                np.resize(d["dhi"], z.size), np.resize(d["dni"], z.size),
                np.resize(extra, z.size), z,
                np.resize(d["solar_azimuth"], z.size),
                np.resize(airmass, z.size))

    parity(lambda z: irradiance.perez(*args(z), return_components=True),
           lambda z: pvlib.irradiance.perez(*args(z), return_components=True),
           d["solar_zenith"], rtol=1e-12)


def test_parallel_clearness_index_and_zenith_independent(big_irr):
    d = big_irr
    extra = pvlib.irradiance.get_extra_radiation(d["doy"])
    parity(lambda z: irradiance.clearness_index(np.resize(d["ghi"], z.size), z,
                                                np.resize(extra, z.size)),
           lambda z: pvlib.irradiance.clearness_index(
               np.resize(d["ghi"], z.size), z, np.resize(extra, z.size)),
           d["solar_zenith"], rtol=1e-12)
    kt = irradiance.clearness_index(tile(d["ghi"]), tile(d["solar_zenith"]),
                                    tile(extra))
    airmass = pv.atmosphere.get_relative_airmass(tile(d["solar_zenith"]))
    ktp = irradiance.clearness_index_zenith_independent(kt, airmass)
    theirs = pvlib.irradiance.clearness_index_zenith_independent(kt, airmass)
    same(ktp[:LONG], theirs[:LONG], rtol=1e-12)


def test_parallel_equation_of_time_spencer71():
    period = np.arange(1.0, 366.0)
    parity(solarposition.equation_of_time_spencer71,
           pvlib.solarposition.equation_of_time_spencer71, period, rtol=1e-12)
    parity(solarposition.declination_spencer71,
           pvlib.solarposition.declination_spencer71, period, rtol=1e-12)


@pytest.mark.parametrize("name, rtol", [
    ("equation_of_time_spencer71", 1e-12),
    ("equation_of_time_pvcdrom", 1e-12),
    ("declination_spencer71", 1e-12),
    ("declination_cooper69", 1e-10),
])
def test_fused_day_angle_kernels_agree_across_the_threshold(name, rtol):
    """The fused day-angle models take one dispatch, not two.

    Below `PAR_MIN_MEDIUM` they run the same `one` in a serial loop and above it
    the same `one` goes to `parallelize`, so both sides must be bit-identical
    and must both match pvlib.
    """
    period = np.arange(1.0, 366.0)
    ours_f = getattr(solarposition, name)
    theirs_f = getattr(pvlib.solarposition, name)

    short = ours_f(period)
    same(short, theirs_f(period), rtol=rtol, atol=1e-12)

    # LONG is above every PAR_MIN_* threshold, so the tiled call dispatches.
    long_period = tile(period, LONG)
    long = ours_f(long_period)
    same(long, theirs_f(long_period), rtol=rtol, atol=1e-12)
    reps = long_period.size // period.size
    same(long, np.tile(short, reps), rtol=0, atol=0)




def test_parallel_dirint_matches_pvlib(big_irr):
    d = big_irr
    n = LONG // d["solar_zenith"].size
    ghi = tile(d["ghi"], n)
    zenith = tile(d["solar_zenith"], n)
    times = pd.DatetimeIndex(np.tile(
        pd.date_range("2020-01-01", periods=d["solar_zenith"].size,
                      freq="h", tz="UTC").to_numpy(),
        n // d["solar_zenith"].size))
    ours = irradiance.dirint(ghi, zenith, times)
    theirs = pvlib.irradiance.dirint(ghi, zenith, times)
    same(ours, theirs, rtol=DECOMP_TOL)


def test_parallel_spa_loop_matches_pvlib_and_short_loop():
    period = pd.date_range("2020-06-21", periods=512, freq="h", tz="UTC")
    times = pd.DatetimeIndex(np.tile(period.to_numpy(), 8))
    assert len(times) > PAR_MIN_TINY
    ours = pv.solarposition.spa_python(times, 35.0, -105.0)
    theirs = pvlib.solarposition.spa_python(times, 35.0, -105.0)
    short = pv.solarposition.spa_python(period, 35.0, -105.0)
    for key in ("zenith", "azimuth", "elevation"):
        same(ours[key].to_numpy(), theirs[key], rtol=1e-9, atol=1e-9)
        same(ours[key].to_numpy(), np.tile(short[key].to_numpy(), 8),
             rtol=1e-12)


def test_parallel_nrel_earthsun_distance():
    period = pd.date_range("2020-06-21", periods=512, freq="h", tz="UTC")
    times = pd.DatetimeIndex(np.tile(period.to_numpy(), 8))
    ours = solarposition.nrel_earthsun_distance(times)
    theirs = pvlib.solarposition.nrel_earthsun_distance(times)
    short = solarposition.nrel_earthsun_distance(period)
    same(ours, theirs, rtol=1e-12)
    same(ours.to_numpy(), np.tile(short.to_numpy(), 8), rtol=1e-12)


def test_parallel_calculate_deltat():
    period = np.array([-600.0, 0.0, 700.0, 1650.0, 1750.0, 1830.0, 1880.0,
                        1910.0, 1930.0, 1950.0, 1970.0, 1990.0, 2020.0,
                        2070.0, 2200.0])
    year = tile(period)
    month = np.full(year.size, 6.0)
    ours = spa.calculate_deltat(year, month)
    theirs = pvlib.spa.calculate_deltat(year, month)
    same(ours, theirs, rtol=1e-12)
    short = spa.calculate_deltat(period, np.full(period.size, 6.0))
    same(ours, np.tile(short, int(np.ceil(ours.size / period.size))),
         rtol=1e-12)


# ------------------------------------------------- broadcasting conventions
@pytest.mark.parametrize("fn", [
    lambda a, b: irradiance.aoi(a, b, 45.0, 180.0),
    lambda a, b: irradiance.beam_component(a, b, 45.0, 180.0, 800.0),
    lambda a, b: irradiance.klucher(a, b, 100.0, 900.0, 45.0, 180.0),
    lambda a, b: irradiance.perez(a, b, 100.0, 800.0, 1366.0, 45.0, 180.0, 1.4),
    lambda a, b: atmosphere.get_absolute_airmass(a, b),
    lambda a, b: solarposition.solar_azimuth_analytical(a, b, 0.2, np.full(5, 0.5)),
])
def test_mismatched_lengths_raise(fn):
    # A kernel handed a short buffer reads past its end, so the mismatch has
    # to be refused here; pvlib refuses it too.
    with pytest.raises(ValueError):
        fn(np.full(5, 30.0), np.full(3, 180.0))


def test_mismatched_doy_lengths_raise(irr):
    with pytest.raises(ValueError):
        irradiance.disc(irr["ghi"], irr["solar_zenith"], np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        irradiance.dirint(irr["ghi"], irr["solar_zenith"], np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        spa.calculate_deltat(np.array([2020.0, 2021.0]), np.array([1.0, 2.0, 3.0]))


def test_length_one_broadcasts(irr):
    # A length-1 argument is a value repeated across the series, as numpy does.
    n = irr["n"]
    one = np.array([30.0])
    same(irradiance.aoi(one, irr["surface_azimuth"], irr["solar_zenith"],
                        irr["solar_azimuth"]),
         pvlib.irradiance.aoi(one, irr["surface_azimuth"], irr["solar_zenith"],
                              irr["solar_azimuth"]), rtol=1e-12)
    same(irradiance.disc(irr["ghi"], irr["solar_zenith"], 172.0)["dni"],
         pvlib.irradiance.disc(irr["ghi"], irr["solar_zenith"], 172.0)["dni"],
         rtol=DECOMP_TOL)
    same(spa.calculate_deltat(2020.0, np.full(n, 6.0)),
         pvlib.spa.calculate_deltat(2020.0, np.full(n, 6.0)), rtol=1e-12)


@pytest.mark.parametrize("fn, theirs", [
    (lambda a: tools.cosd(a), lambda a: pvlib.tools.cosd(a)),
    (lambda a: tools.sind(a), lambda a: pvlib.tools.sind(a)),
    (lambda a: tools.asind(a), lambda a: pvlib.tools.asind(a)),
    (lambda a: atmosphere.pres2alt(a), lambda a: pvlib.atmosphere.pres2alt(a)),
    (lambda a: atmosphere.get_relative_airmass(a),
     lambda a: pvlib.atmosphere.get_relative_airmass(a)),
    (lambda a: irradiance.get_extra_radiation(a),
     lambda a: pvlib.irradiance.get_extra_radiation(a)),
    (lambda a: irradiance.clearsky_index(a, 800.0),
     lambda a: pvlib.irradiance.clearsky_index(a, 800.0)),
    (lambda a: irradiance.aoi(a, 180.0, 45.0, 180.0),
     lambda a: pvlib.irradiance.aoi(a, 180.0, 45.0, 180.0)),
    (lambda a: solarposition.declination_cooper69(a),
     lambda a: pvlib.solarposition.declination_cooper69(a)),
])
def test_zero_d_input_returns_scalar(fn, theirs):
    # A 0-d array in, a float out, which is what numpy's ufuncs do.
    assert np.ndim(fn(np.array(45.0))) == 0
    same(fn(np.array(45.0)), theirs(np.array(45.0)), rtol=1e-9, atol=1e-9)
    assert np.ndim(fn(np.float64(45.0))) == 0


def test_perez_coefficient_table_nan_row_is_all_nan():
    # Row 0 stands in for upstream's appended all-NaN row, so all three
    # coefficients have to be NaN, not just the first.
    f1, f2 = pv._lib.perez_coefficient_table("allsitescomposite1990")
    assert np.all(np.isnan(f1[0]))
    assert np.all(np.isnan(f2[0]))
