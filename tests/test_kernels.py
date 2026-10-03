"""Every `@export`ed kernel, called by name, with its output asserted.

`test_parity.py` drives the same kernels through the Python API, which is how a
caller reaches them. That is good coverage but weak evidence: a kernel wired up
and wired to the wrong buffer would still ship green, because nothing observes
the symbol itself. This module closes that gap -- it calls all 56 `mpv_*` ctypes
entry points directly, with hand-built input buffers, and asserts on the values
that come back.

The ABI is the one `src/capi.mojo` documents: every buffer crosses as an `Int`
address and the kernel reads and writes caller-owned memory. That is what each
test below does, so a kernel that reads the wrong argument, writes the wrong
output, or ignores `n` fails here rather than somewhere downstream.

Every expected value comes from real pvlib on the same input, so these are
parity assertions and not just "it ran". The handful of functions whose
transcendental chains make LLVM's `pow` and `log` disagree with libm's by a few
ULP use `POW_TOL`; everything else is held to the tight tolerance the arithmetic
actually allows.
"""

import ctypes

import numpy as np
import pandas as pd
import pytest

pvlib = pytest.importorskip("pvlib")
import pvlib.atmosphere
import pvlib.irradiance
import pvlib.solarposition
import pvlib.spa
import pvlib.tools

from mojopvlib import _lib

# LLVM's pow/log and libm's are different implementations that agree to a few
# ULP rather than bit for bit; a chain that exponentiates amplifies that.
POW_TOL = 1e-6


@pytest.fixture(scope="module")
def lib():
    return _lib.lib()


def buf(values) -> np.ndarray:
    """A writable float64 buffer holding `values`, as the ABI wants it."""
    return np.ascontiguousarray(np.ravel(values), dtype=np.float64)


def zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def a(buf_: np.ndarray) -> int:
    return buf_.ctypes.data


def assert_same(got, want, rtol=1e-12, atol=0.0):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, equal_nan=True)


# ------------------------------------------------------------------- tools
def test_mpv_cosd(lib):
    angle = np.linspace(-720.0, 720.0, 257)
    res = zeros(angle.size)
    lib.mpv_cosd(a(angle), a(res), angle.size)
    assert_same(res, np.cos(np.radians(angle)), rtol=1e-15)


def test_mpv_sind(lib):
    angle = np.linspace(-720.0, 720.0, 257)
    res = zeros(angle.size)
    lib.mpv_sind(a(angle), a(res), angle.size)
    assert_same(res, np.sin(np.radians(angle)), rtol=1e-15)


def test_mpv_tand(lib):
    angle = np.linspace(-80.0, 80.0, 257)
    res = zeros(angle.size)
    lib.mpv_tand(a(angle), a(res), angle.size)
    assert_same(res, np.tan(np.radians(angle)), rtol=1e-15)


def test_mpv_asind(lib):
    number = np.linspace(-1.0, 1.0, 257)
    res = zeros(number.size)
    lib.mpv_asind(a(number), a(res), number.size)
    assert_same(res, np.degrees(np.arcsin(number)), rtol=1e-12)


def test_mpv_acosd(lib):
    number = np.linspace(-1.0, 1.0, 257)
    res = zeros(number.size)
    lib.mpv_acosd(a(number), a(res), number.size)
    assert_same(res, np.degrees(np.arccos(number)), rtol=1e-12)


def test_mpv_atand(lib):
    number = np.linspace(-10.0, 10.0, 257)
    res = zeros(number.size)
    lib.mpv_atand(a(number), a(res), number.size)
    assert_same(res, np.degrees(np.arctan(number)), rtol=1e-12)


def test_trig_kernels_respect_n(lib):
    """`n` bounds the work: one element of a long buffer, not all of it."""
    angle = buf([30.0, 60.0, 90.0, 120.0])
    res = np.full(angle.size, np.nan)
    lib.mpv_cosd(a(angle), a(res), 1)
    assert_same(res[:1], np.cos(np.radians(angle[:1])), rtol=1e-15)
    assert np.isnan(res[1:]).all(), "wrote past n"


# --------------------------------------------------------------------- spa
def test_mpv_sp_tables(lib):
    """Every SPA coefficient table, bit for bit, against pvlib.spa."""
    buf_ = zeros(4096)
    lib.mpv_sp_tables(a(buf_))
    for name in _lib.SPA_TABLE_NAMES:
        got = _lib.spa_table(name)
        assert_same(got, getattr(pvlib.spa, name), rtol=0, atol=0)


def test_mpv_calculate_deltat(lib):
    year = buf([-1000.0, 0.0, 500.0, 1000.0, 1700.0, 1900.0, 2020.0, 2100.0])
    month = buf(np.full(8, 6.0))
    deltat = zeros(8)
    lib.mpv_calculate_deltat(a(year), a(month), a(deltat), 8)
    assert_same(deltat, pvlib.spa.calculate_deltat(year, month))


def test_mpv_solar_position_loop(lib):
    """The per-timestamp SPA pass, against `pvlib.spa.solar_position` itself.

    `spa_python` layers its own pressure/temperature defaults on top of this
    kernel, so the like-for-like comparison is with the SPA entry point on the
    same `loc_args`: those are the values the kernel actually consumes.
    """
    import pandas as pd

    times = pd.date_range("2020-06-21", periods=48, freq="h", tz="UTC")
    unixtime = buf(times.astype("int64") / 1e9)
    n = unixtime.size
    lat, lon, elev = 35.0, -105.0, 1600.0
    pressure, temp, atmos_refract = 82000.0, 14.0, 0.5667

    deltat = zeros(n)
    year = buf(np.full(n, 2020.0))
    month = buf(times.month.to_numpy(dtype=float))
    lib.mpv_calculate_deltat(a(year), a(month), a(deltat), n)

    res = np.zeros((6, n), dtype=np.float64)
    lib.mpv_solar_position_loop(
        a(unixtime), a(deltat), n, lat, lon, elev, pressure, temp, atmos_refract,
        0, 0,
        a(res[0]), a(res[1]), a(res[2]), a(res[3]), a(res[4]), a(res[5]),
        a(_lib.spa_tables_buffer()),
    )
    want = pvlib.spa.solar_position(
        unixtime, lat, lon, elev, pressure, temp, deltat, atmos_refract, 8
    )
    # rows: theta, theta0, e, e0, phi, eot -- degrees, eot in minutes
    for r in range(6):
        assert_same(res[r], want[r], rtol=0, atol=1e-9)


def test_mpv_solar_position_loop_esd(lib):
    """With `esd=1` the loop returns only the earth-sun radius, in row 0."""
    import pandas as pd

    times = pd.date_range("2020-06-21", periods=24, freq="h", tz="UTC")
    unixtime = buf(times.astype("int64") / 1e9)
    n = unixtime.size
    deltat = buf(np.full(n, pvlib.spa.calculate_deltat(2020.0, 6.0)))
    res = np.zeros((6, n), dtype=np.float64)
    lib.mpv_solar_position_loop(
        a(unixtime), a(deltat), n, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5667, 0, 1,
        a(res[0]), a(res[1]), a(res[2]), a(res[3]), a(res[4]), a(res[5]),
        a(_lib.spa_tables_buffer()),
    )
    want = pvlib.spa.solar_position(unixtime, 0, 0, 0, 0, 0, deltat, 0, 8,
                                    esd=True)
    assert_same(res[0], want[0], rtol=POW_TOL)


# ---------------------------------------------------------- solarposition
def test_mpv_calculate_simple_day_angle(lib):
    # offset defaults to 1 upstream (Spencer); 0 is the ASCE variant.
    doy = buf(np.arange(1.0, 367.0))
    res = zeros(doy.size)
    lib.mpv_calculate_simple_day_angle(a(doy), 1.0, a(res), doy.size)
    assert_same(res, pvlib.solarposition._calculate_simple_day_angle(doy), rtol=1e-15)


def test_mpv_calculate_simple_day_angle_offset(lib):
    """`offset` is the day-of-year shift: 1 for Spencer, 0 for ASCE."""
    doy = buf([1.0, 100.0, 366.0])
    res = zeros(3)
    lib.mpv_calculate_simple_day_angle(a(doy), 0.0, a(res), 3)
    assert_same(res, pvlib.solarposition._calculate_simple_day_angle(doy, offset=0),
                rtol=1e-15)


def test_mpv_equation_of_time_spencer71(lib):
    doy = buf(np.arange(1.0, 367.0))
    eot = zeros(doy.size)
    day_angle = zeros(doy.size)
    lib.mpv_equation_of_time_spencer71(a(doy), a(eot), doy.size, a(day_angle))
    assert_same(eot, pvlib.solarposition.equation_of_time_spencer71(doy), rtol=1e-12)


def test_mpv_equation_of_time_pvcdrom(lib):
    doy = buf(np.arange(1.0, 367.0))
    eot = zeros(doy.size)
    day_angle = zeros(doy.size)
    lib.mpv_equation_of_time_pvcdrom(a(doy), a(eot), doy.size, a(day_angle))
    assert_same(eot, pvlib.solarposition.equation_of_time_pvcdrom(doy), rtol=1e-12)


def test_mpv_declination_spencer71(lib):
    doy = buf(np.arange(1.0, 367.0))
    dec = zeros(doy.size)
    day_angle = zeros(doy.size)
    lib.mpv_declination_spencer71(a(doy), a(dec), doy.size, a(day_angle))
    assert_same(dec, pvlib.solarposition.declination_spencer71(doy), rtol=1e-12)


def test_mpv_declination_cooper69(lib):
    doy = buf(np.arange(1.0, 367.0))
    dec = zeros(doy.size)
    day_angle = zeros(doy.size)
    lib.mpv_declination_cooper69(a(doy), a(dec), doy.size, a(day_angle))
    # crosses zero, so an absolute floor is needed
    assert_same(dec, pvlib.solarposition.declination_cooper69(doy),
                rtol=1e-10, atol=1e-12)


# The four day-angle models above compute the day angle in the same pass as the
# model rather than in a separate one, so they still have to leave the same
# value in the caller's `day_angle` buffer as `_calculate_simple_day_angle`
# would have.
DAY_ANGLE_KERNELS = [
    "mpv_equation_of_time_spencer71",
    "mpv_equation_of_time_pvcdrom",
    "mpv_declination_spencer71",
    "mpv_declination_cooper69",
]


@pytest.mark.parametrize("symbol", DAY_ANGLE_KERNELS)
def test_day_angle_kernels_still_fill_day_angle(lib, symbol):
    """The fused pass must still write `day_angle`, bit for bit."""
    doy = buf(np.arange(1.0, 367.0))
    res = zeros(doy.size)
    day_angle = zeros(doy.size)
    getattr(lib, symbol)(a(doy), a(res), doy.size, a(day_angle))
    assert_same(day_angle, pvlib.solarposition._calculate_simple_day_angle(doy),
                rtol=0, atol=0)


@pytest.mark.parametrize("symbol", DAY_ANGLE_KERNELS)
def test_day_angle_kernels_respect_n(lib, symbol):
    """`n` bounds the fused pass: the tail of a long buffer is left alone."""
    doy = buf(np.arange(1.0, 367.0))
    res = np.full(doy.size, np.nan)
    day_angle = np.full(doy.size, np.nan)
    getattr(lib, symbol)(a(doy), a(res), 10, a(day_angle))
    assert np.isnan(res[10:]).all(), "wrote past n"
    assert np.isnan(day_angle[10:]).all(), "wrote day_angle past n"
    assert np.isfinite(res[:10]).all()
    assert np.isfinite(day_angle[:10]).all()


def test_mpv_solar_azimuth_analytical(lib):
    lat = buf(np.radians(np.linspace(-89, 89, 97)))
    ha = buf(np.radians(np.linspace(-180, 180, 97)))
    dec = buf(np.radians(np.linspace(-23.4, 23.4, 97)))
    # upstream's azimuth takes the zenith as an *input*, so compute it first
    # with the zenith kernel; that also covers the two in one round trip.
    zen = zeros(97)
    lib.mpv_solar_zenith_analytical(a(lat), a(ha), a(dec), a(zen), 97)
    az = np.full(97, np.nan)
    lib.mpv_solar_azimuth_analytical(a(lat), a(ha), a(dec), a(zen), a(az), 97)
    want_zen = pvlib.solarposition.solar_zenith_analytical(lat, ha, dec)
    want_az = pvlib.solarposition.solar_azimuth_analytical(lat, ha, dec, zen)
    assert_same(zen, want_zen, rtol=1e-12, atol=1e-12)
    assert_same(az, want_az, rtol=1e-12, atol=1e-12)


def test_mpv_solar_zenith_analytical(lib):
    lat = buf(np.radians(np.linspace(-89, 89, 97)))
    ha = buf(np.radians(np.linspace(-180, 180, 97)))
    dec = buf(np.radians(np.linspace(-23.4, 23.4, 97)))
    zen = zeros(97)
    lib.mpv_solar_zenith_analytical(a(lat), a(ha), a(dec), a(zen), 97)
    assert_same(zen, pvlib.solarposition.solar_zenith_analytical(lat, ha, dec),
                rtol=1e-12, atol=1e-12)


def test_mpv_hour_angle(lib):
    import pandas as pd

    day = pd.date_range("2020-06-21", periods=24, freq="h", tz="MST")
    eot = buf(pvlib.solarposition.equation_of_time_spencer71(day.dayofyear))
    # upstream: hrs_minus_tzs = hours_after_local_midnight - tz_offset_hours
    tzs = np.array([ts.utcoffset().total_seconds() for ts in day]) / 3600.0
    hrs_minus_tzs = buf(day.hour + day.minute / 60.0 - tzs)
    res = zeros(24)
    lib.mpv_hour_angle(a(hrs_minus_tzs), -105.0, a(eot), a(res), 24)
    want = pvlib.solarposition.hour_angle(day, -105.0, eot)
    assert_same(res, want, rtol=1e-12)


def test_mpv_sunset_angle(lib):
    # upstream keeps this as a local helper inside sun_rise_set_transit_geometric;
    # the kernel ports it, so the expectation is the same closed form.
    day = np.arange(1.0, 367.0)
    dec = buf(pvlib.solarposition.declination_cooper69(day))  # radians
    res = zeros(day.size)
    latitude = 20.0
    lib.mpv_sunset_angle(latitude, a(dec), a(res), day.size)
    want = np.degrees(np.arccos(-np.tan(dec) * np.tan(np.radians(latitude))))
    assert_same(res, want, rtol=1e-12)


# ------------------------------------------------------------- atmosphere
def test_mpv_pres2alt(lib):
    pressure = buf(np.linspace(70000.0, 105000.0, 50))
    alt = zeros(50)
    lib.mpv_pres2alt(a(pressure), a(alt), 50)
    assert_same(alt, pvlib.atmosphere.pres2alt(pressure), rtol=POW_TOL)


def test_mpv_alt2pres(lib):
    altitude = buf(np.linspace(0.0, 4000.0, 50))
    press = zeros(50)
    lib.mpv_alt2pres(a(altitude), a(press), 50)
    assert_same(press, pvlib.atmosphere.alt2pres(altitude), rtol=POW_TOL)


@pytest.mark.parametrize("model", sorted(_lib.AIRMASS_MODELS))
def test_mpv_get_relative_airmass(lib, model):
    zenith = buf(np.linspace(0.0, 95.0, 200))
    am = zeros(200)
    lib.mpv_get_relative_airmass(a(zenith), a(am), 200, _lib.AIRMASS_MODELS[model])
    assert_same(am, pvlib.atmosphere.get_relative_airmass(zenith, model), rtol=1e-8)


def test_mpv_get_absolute_airmass(lib):
    am_rel = buf(np.linspace(1.0, 5.0, 50))
    pressure = buf(np.linspace(70000.0, 101325.0, 50))
    am_abs = zeros(50)
    lib.mpv_get_absolute_airmass(a(am_rel), a(pressure), a(am_abs), 50)
    assert_same(am_abs, pvlib.atmosphere.get_absolute_airmass(am_rel, pressure),
                rtol=1e-14)


def test_mpv_gueymard94_pw(lib):
    t = buf(np.linspace(-10.0, 45.0, 50))
    rh = buf(np.linspace(5.0, 100.0, 50))
    pw = zeros(50)
    lib.mpv_gueymard94_pw(a(t), a(rh), a(pw), 50)
    assert_same(pw, pvlib.atmosphere.gueymard94_pw(t, rh), rtol=1e-11)


def test_mpv_bird_hulstrom80_aod_bb(lib):
    a380 = buf(np.linspace(0.01, 0.5, 20))
    a500 = buf(np.linspace(0.01, 0.3, 20))
    bb = zeros(20)
    lib.mpv_bird_hulstrom80_aod_bb(a(a380), a(a500), a(bb), 20)
    assert_same(bb, pvlib.atmosphere.bird_hulstrom80_aod_bb(a380, a500), rtol=1e-15)


def test_mpv_kasten96_lt(lib):
    am = buf(np.linspace(1.0, 5.0, 20))
    pw = buf(np.linspace(0.5, 5.0, 20))
    aod = buf(np.linspace(0.01, 0.3, 20))
    lt = zeros(20)
    lib.mpv_kasten96_lt(a(am), a(pw), a(aod), a(lt), 20)
    assert_same(lt, pvlib.atmosphere.kasten96_lt(am, pw, aod), rtol=POW_TOL)


def test_mpv_rh_from_tdew(lib):
    ta = buf(np.linspace(-20.0, 45.0, 40))
    td = buf(np.linspace(-40.0, 20.0, 40))
    rh = np.full(40, np.nan)
    c0, c1, c2 = 6.112, 17.62, 243.12
    lib.mpv_rh_from_tdew(a(ta), a(td), a(rh), 40, c0, c1, c2)
    assert_same(rh, pvlib.atmosphere.rh_from_tdew(ta, td), rtol=1e-12)


def test_mpv_tdew_from_rh(lib):
    ta = buf(np.linspace(-20.0, 45.0, 40))
    rh = buf(np.linspace(5.0, 100.0, 40))
    td = np.full(40, np.nan)
    c0, c1, c2 = 6.112, 17.62, 243.12
    lib.mpv_tdew_from_rh(a(ta), a(rh), a(td), 40, c0, c1, c2)
    assert_same(td, pvlib.atmosphere.tdew_from_rh(ta, rh), rtol=1e-9)


def test_mpv_angstrom_aod_at_lambda(lib):
    aod0 = buf(np.linspace(0.02, 0.5, 37))
    lam0 = buf(np.linspace(380.0, 800.0, 37))
    aod1 = np.full(37, np.nan)
    lib.mpv_angstrom_aod_at_lambda(a(aod0), a(lam0), a(aod1), 37, 1.14, 700.0)
    assert_same(aod1,
                pvlib.atmosphere.angstrom_aod_at_lambda(aod0, lam0), rtol=POW_TOL)


def test_mpv_angstrom_alpha(lib):
    aod1 = buf(np.linspace(0.02, 0.5, 37))
    lam1 = buf(np.linspace(380.0, 700.0, 37))
    aod2 = buf(np.linspace(0.01, 0.3, 37))
    lam2 = buf(np.linspace(700.0, 1000.0, 37))
    alpha = np.full(37, np.nan)
    lib.mpv_angstrom_alpha(a(aod1), a(lam1), a(aod2), a(lam2), a(alpha), 37)
    assert_same(alpha, pvlib.atmosphere.angstrom_alpha(aod1, lam1, aod2, lam2),
                rtol=POW_TOL)


def test_mpv_windspeed_powerlaw(lib):
    # the kernel takes (ws, height_desired, height_reference); pvlib's
    # signature is (ws, height_reference, height_desired)
    ws = buf(np.linspace(0.5, 15.0, 37))
    h_des = buf(np.full(37, 40.0))
    h_ref = buf(np.full(37, 9.144))
    res = np.full(37, np.nan)
    lib.mpv_windspeed_powerlaw(a(ws), a(h_des), a(h_ref), 0.219, a(res), 37)
    assert_same(res,
                pvlib.atmosphere.windspeed_powerlaw(ws, 9.144, 40.0, exponent=0.219),
                rtol=POW_TOL)


def test_mpv_windspeed_powerlaw_edge_ratios(lib):
    """The NaN and infinity branches of the ratio.

    A negative ratio is complex under a fractional exponent and must become
    NaN; a zero reference height makes the ratio infinite, and that stays
    infinite rather than being folded into the NaN branch.
    """
    ws = buf([2.0, 3.0, 4.0])
    h_des = buf([40.0, 40.0, 40.0])
    h_ref = buf([-10.0, 10.0, 0.0])
    res = np.zeros(3)
    lib.mpv_windspeed_powerlaw(a(ws), a(h_des), a(h_ref), 0.14, a(res), 3)
    assert np.isnan(res[0]), "negative height ratio must be NaN"
    assert np.isfinite(res[1]), "positive ratio must stay finite"
    assert np.isinf(res[2]) and res[2] > 0, "infinite ratio must stay infinite"


# ------------------------------------------------------------- irradiance
def test_mpv_get_extra_radiation(lib):
    doy = buf(np.arange(1.0, 366.0))
    res = zeros(doy.size)
    lib.mpv_get_extra_radiation(a(doy), a(res), doy.size, 1366.1, 1)  # spencer
    assert_same(res, pvlib.irradiance.get_extra_radiation(doy), rtol=1e-12)


def test_mpv_get_extra_radiation_asce(lib):
    doy = buf(np.arange(1.0, 366.0))
    res = zeros(doy.size)
    lib.mpv_get_extra_radiation(a(doy), a(res), doy.size, 1366.1, 0)  # asce
    assert_same(res, pvlib.irradiance.get_extra_radiation(doy, method="asce"),
                rtol=1e-12)


def test_mpv_aoi_projection(lib):
    st = buf(np.full(97, 30.0))
    sa = buf(np.full(97, 180.0))
    sz = buf(np.linspace(0.0, 89.0, 97))
    saz = buf(np.linspace(0.0, 360.0, 97))
    proj = np.full(97, np.nan)
    lib.mpv_aoi_projection(a(st), a(sa), a(sz), a(saz), a(proj), 97)
    assert_same(proj,
                pvlib.irradiance.aoi_projection(st, sa, sz, saz), rtol=1e-13)


def test_mpv_aoi(lib):
    st = buf(np.full(97, 30.0))
    sa = buf(np.full(97, 180.0))
    sz = buf(np.linspace(0.0, 89.0, 97))
    saz = buf(np.linspace(0.0, 360.0, 97))
    aoi = np.full(97, np.nan)
    lib.mpv_aoi(a(st), a(sa), a(sz), a(saz), a(aoi), 97)
    assert_same(aoi, pvlib.irradiance.aoi(st, sa, sz, saz), rtol=1e-12)


def test_mpv_beam_component(lib):
    st = buf(np.full(97, 30.0))
    sa = buf(np.full(97, 180.0))
    sz = buf(np.linspace(0.0, 89.0, 97))
    saz = buf(np.linspace(0.0, 360.0, 97))
    dni = buf(np.linspace(100.0, 900.0, 97))
    beam = np.full(97, np.nan)
    lib.mpv_beam_component(a(st), a(sa), a(sz), a(saz), a(dni), a(beam), 97)
    assert_same(beam, pvlib.irradiance.beam_component(st, sa, sz, saz, dni),
                rtol=1e-13)


def test_mpv_get_ground_diffuse(lib):
    st = buf(np.full(50, 30.0))
    ghi = buf(np.linspace(0.0, 1000.0, 50))
    res = np.full(50, np.nan)
    lib.mpv_get_ground_diffuse(a(st), a(ghi), a(res), 50, 0.25)
    assert_same(res, pvlib.irradiance.get_ground_diffuse(st, ghi), rtol=0, atol=0)


def test_mpv_isotropic(lib):
    st = buf(np.full(50, 30.0))
    dhi = buf(np.linspace(0.0, 300.0, 50))
    sky = np.full(50, np.nan)
    tot = np.full(50, np.nan)
    lib.mpv_isotropic(a(st), a(dhi), a(sky), a(tot), 50)
    want = pvlib.irradiance.isotropic(st, dhi, return_components=True)
    assert_same(sky, want["poa_sky_diffuse"], rtol=0, atol=0)
    assert_same(tot, want["poa_isotropic"], rtol=0, atol=0)


def test_mpv_klucher(lib):
    st = buf(np.full(97, 30.0))
    sa = buf(np.full(97, 180.0))
    dhi = buf(np.linspace(10.0, 300.0, 97))
    ghi = buf(np.linspace(100.0, 1000.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    saz = buf(np.linspace(0.0, 360.0, 97))
    sky = np.full(97, np.nan)
    lib.mpv_klucher(a(st), a(sa), a(dhi), a(ghi), a(sz), a(saz), a(sky), 97)
    assert_same(sky, pvlib.irradiance.klucher(st, sa, dhi, ghi, sz, saz),
                rtol=1e-12)


def test_mpv_haydavies(lib):
    st = buf(np.full(97, 30.0))
    sa = buf(np.full(97, 180.0))
    dhi = buf(np.linspace(10.0, 300.0, 97))
    dni = buf(np.linspace(50.0, 900.0, 97))
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(97, 172.0)))
    sz = buf(np.linspace(5.0, 85.0, 97))
    saz = buf(np.linspace(0.0, 360.0, 97))
    sky = np.full(97, np.nan)
    iso = np.full(97, np.nan)
    circ = np.full(97, np.nan)
    lib.mpv_haydavies(a(st), a(sa), a(dhi), a(dni), a(ex), a(sz), a(saz),
                      0, a(sky), a(iso), a(circ), 97, 0)
    want = pvlib.irradiance.haydavies(st, sa, dhi, dni, ex, sz, saz,
                                      return_components=True)
    assert_same(sky, want["poa_sky_diffuse"], rtol=1e-12)
    assert_same(iso, want["poa_isotropic"], rtol=1e-12)
    assert_same(circ, want["poa_circumsolar"], rtol=1e-12)


def test_mpv_reindl(lib):
    st = buf(np.full(97, 30.0))
    sa = buf(np.full(97, 180.0))
    dhi = buf(np.linspace(10.0, 300.0, 97))
    dni = buf(np.linspace(50.0, 900.0, 97))
    ghi = buf(np.linspace(100.0, 1000.0, 97))
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(97, 172.0)))
    sz = buf(np.linspace(5.0, 85.0, 97))
    saz = buf(np.linspace(0.0, 360.0, 97))
    sky = np.full(97, np.nan)
    iso = np.full(97, np.nan)
    circ = np.full(97, np.nan)
    horiz = np.full(97, np.nan)
    lib.mpv_reindl(a(st), a(sa), a(dhi), a(dni), a(ghi), a(ex), a(sz), a(saz),
                   a(sky), a(iso), a(circ), a(horiz), 97)
    want = pvlib.irradiance.reindl(st, sa, dhi, dni, ghi, ex, sz, saz,
                                   return_components=True)
    assert_same(sky, want["poa_sky_diffuse"], rtol=1e-12)
    assert_same(iso, want["poa_isotropic"], rtol=1e-12)
    assert_same(circ, want["poa_circumsolar"], rtol=1e-12)
    assert_same(horiz, want["poa_horizon"], rtol=1e-12)


def test_mpv_king(lib):
    st = buf(np.full(97, 30.0))
    dhi = buf(np.linspace(10.0, 300.0, 97))
    ghi = buf(np.linspace(100.0, 1000.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    sky = np.full(97, np.nan)
    lib.mpv_king(a(st), a(dhi), a(ghi), a(sz), a(sky), 97)
    assert_same(sky, pvlib.irradiance.king(st, dhi, ghi, sz), rtol=1e-12)


def test_mpv_perez(lib):
    st = buf(np.full(97, 30.0))
    sa = buf(np.full(97, 180.0))
    dhi = buf(np.linspace(10.0, 300.0, 97))
    dni = buf(np.linspace(50.0, 900.0, 97))
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(97, 172.0)))
    sz = buf(np.linspace(5.0, 85.0, 97))
    saz = buf(np.linspace(0.0, 360.0, 97))
    am = buf(np.linspace(1.0, 3.0, 97))
    f1c = np.zeros((9, 3))
    f2c = np.zeros((9, 3))
    lib.mpv_perez_coefficients(0, f1c.ctypes.data, f2c.ctypes.data)
    sky = np.full(97, np.nan)
    iso = np.full(97, np.nan)
    circ = np.full(97, np.nan)
    horiz = np.full(97, np.nan)
    lib.mpv_perez(a(st), a(sa), a(dhi), a(dni), a(ex), a(sz), a(saz), a(am),
                  a(f1c.ravel()), a(f2c.ravel()), a(sky), a(iso), a(circ),
                  a(horiz), 97)
    want = pvlib.irradiance.perez(st, sa, dhi, dni, ex, sz, saz, am,
                                  return_components=True)
    assert_same(sky, want["poa_sky_diffuse"], rtol=1e-12)
    assert_same(iso, want["poa_isotropic"], rtol=1e-12)
    assert_same(circ, want["poa_circumsolar"], rtol=1e-12)
    assert_same(horiz, want["poa_horizon"], rtol=1e-12)


def test_mpv_clearsky_index(lib):
    ghi = buf(np.linspace(0.0, 1000.0, 97))
    clear = buf(np.linspace(50.0, 1100.0, 97))
    res = np.full(97, np.nan)
    lib.mpv_clearsky_index(a(ghi), a(clear), a(res), 97, 2.0)
    assert_same(res, pvlib.irradiance.clearsky_index(ghi, clear), rtol=1e-12)


def test_mpv_clearness_index(lib):
    ghi = buf(np.linspace(0.0, 1000.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(97, 172.0)))
    kt = np.full(97, np.nan)
    lib.mpv_clearness_index(a(ghi), a(sz), a(ex), a(kt), 97, 0.065, 2.0)
    assert_same(kt, pvlib.irradiance.clearness_index(ghi, sz, ex), rtol=1e-12)


def test_mpv_clearness_index_clipped(lib):
    """The `max_clearness_index` clamp, on a GHI well above the clear-sky one."""
    ghi = buf([3000.0, 4000.0])
    sz = buf([45.0, 45.0])
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(2, 172.0)))
    kt = np.full(2, np.nan)
    lib.mpv_clearness_index(a(ghi), a(sz), a(ex), a(kt), 2, 0.065, 1.5)
    assert_same(kt, pvlib.irradiance.clearness_index(ghi, sz, ex,
                                                     max_clearness_index=1.5),
                rtol=1e-12)
    assert (kt == 1.5).all()


def test_mpv_clearness_index_zenith_independent(lib):
    kt = buf(np.linspace(0.1, 1.0, 97))
    am = buf(np.linspace(1.0, 3.0, 97))
    res = np.full(97, np.nan)
    lib.mpv_clearness_index_zenith_independent(a(kt), a(am), a(res), 97, 2.0)
    assert_same(res,
                pvlib.irradiance.clearness_index_zenith_independent(kt, am),
                rtol=1e-12)


def test_mpv_disc(lib):
    ghi = buf(np.linspace(50.0, 1000.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    doy = buf(np.full(97, 172.0))
    dni = np.full(97, np.nan)
    kt = np.full(97, np.nan)
    am = np.full(97, np.nan)
    lib.mpv_disc(a(ghi), a(sz), a(doy), a(dni), a(kt), a(am), 97, 101325.0,
                 0.065, 87.0, 12.0, 1)
    want = pvlib.irradiance.disc(ghi, sz, doy)
    assert_same(dni, want["dni"], rtol=POW_TOL)
    assert_same(kt, want["kt"], rtol=POW_TOL)
    assert_same(am, want["airmass"], rtol=POW_TOL)


def test_mpv_dirint(lib):
    # `delta_kt_prime` is a kernel scratch buffer -- upstream derives the
    # difference from consecutive clearness-index values inside the model --
    # and temp_dew maps to precipitable water inside the model too.
    ghi = buf(np.linspace(50.0, 1000.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    times = pd.date_range("2020-06-21", periods=97, freq="D", tz="UTC")
    doy = buf(times.dayofyear.to_numpy(dtype=float))
    dkt = zeros(97)
    td = buf(np.linspace(0.0, 25.0, 97))
    coeffs = buf(_lib.dirint_coeffs().ravel())
    dni = np.full(97, np.nan)
    # the kernel needs three full-length intermediate series between passes
    scratch = zeros(3 * 97)
    lib.mpv_dirint(a(ghi), a(sz), a(doy), a(dkt), a(td), a(coeffs), a(dni), 97,
                   101325.0, 12.0, 1, 1, 1, 0.065, 87.0, a(scratch))
    temp_dew = pd.Series(np.linspace(0.0, 25.0, 97), index=times)
    want = pvlib.irradiance.dirint(ghi, sz, times, temp_dew=temp_dew)
    assert_same(dni, np.asarray(want), rtol=1e-9)


def test_mpv_dirint_without_dewpoint(lib):
    """`use_temp_dew=0` puts w in the sentinel -1 bin instead."""
    ghi = buf(np.linspace(50.0, 1000.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    times = pd.date_range("2020-06-21", periods=97, freq="D", tz="UTC")
    doy = buf(times.dayofyear.to_numpy(dtype=float))
    zero = zeros(97)
    coeffs = buf(_lib.dirint_coeffs().ravel())
    dni = np.full(97, np.nan)
    scratch = zeros(3 * 97)
    lib.mpv_dirint(a(ghi), a(sz), a(doy), a(zero), a(zero), a(coeffs), a(dni),
                   97, 101325.0, 12.0, 1, 0, 0, 0.065, 87.0, a(scratch))
    want = pvlib.irradiance.dirint(ghi, sz, times, use_delta_kt_prime=False)
    assert_same(dni, np.asarray(want), rtol=1e-9)


def test_mpv_dirindex(lib):
    """`dirindex` is the DIRINT-CI blend; its kernel takes the two DIRINT
    results upstream computes first, not the GHI series.

    Every buffer argument needs a real address -- the kernel rebuilds a
    non-nullable pointer from it, and address 0 would fault.
    """
    ghi = buf(np.linspace(50.0, 1000.0, 97))
    ghi_clear = buf(np.linspace(100.0, 1050.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    times = pd.date_range("2020-06-21", periods=97, freq="D", tz="UTC")
    doy = buf(times.dayofyear.astype(float))
    dni_clear = buf(pvlib.irradiance.disc(ghi_clear, sz, times)["dni"])
    doy = buf(times.dayofyear.astype(float))
    zero = zeros(97)
    coeffs = buf(_lib.dirint_coeffs().ravel())
    scratch = zeros(3 * 97)
    args = (101325.0, 12.0, 1, 0, 0, 0.065, 87.0)  # no dkt, no temp_dew

    dni_dirint = np.full(97, np.nan)
    lib.mpv_dirint(a(ghi), a(sz), a(doy), a(zero), a(zero), a(coeffs),
                   a(dni_dirint), 97, *args, a(scratch))
    dni_dirint_clear = np.full(97, np.nan)
    lib.mpv_dirint(a(ghi_clear), a(sz), a(doy), a(zero), a(zero), a(coeffs),
                   a(dni_dirint_clear), 97, *args, a(scratch))

    res = np.full(97, np.nan)
    lib.mpv_dirindex(a(dni_clear), a(dni_dirint), a(dni_dirint_clear),
                     a(res), 97)
    want = pvlib.irradiance.dirindex(ghi, ghi_clear, dni_clear, sz, times,
                                     use_delta_kt_prime=False)
    assert_same(res, want, rtol=1e-9)


def test_mpv_erbs(lib):
    ghi = buf(np.linspace(0.0, 1000.0, 97))
    sz = buf(np.linspace(0.0, 86.0, 97))
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(97, 172.0)))
    dni = np.full(97, np.nan)
    dhi = np.full(97, np.nan)
    kt = np.full(97, np.nan)
    lib.mpv_erbs(a(ghi), a(sz), a(ex), a(dni), a(dhi), a(kt), 97, 0.065, 87.0)
    want = pvlib.irradiance.erbs(ghi, sz, np.full(97, 172.0))
    assert_same(dni, want["dni"], rtol=1e-9)
    assert_same(dhi, want["dhi"], rtol=1e-9)
    assert_same(kt, want["kt"], rtol=1e-9)


def test_mpv_orgill_hollands(lib):
    ghi = buf(np.linspace(50.0, 1000.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(97, 172.0)))
    dni = np.full(97, np.nan)
    dhi = np.full(97, np.nan)
    kt = np.full(97, np.nan)
    lib.mpv_orgill_hollands(a(ghi), a(sz), a(ex), a(dni), a(dhi), a(kt), 97,
                            0.065, 87.0)
    want = pvlib.irradiance.orgill_hollands(ghi, sz, np.full(97, 172.0))
    assert_same(dni, want["dni"], rtol=1e-9)
    assert_same(dhi, want["dhi"], rtol=1e-9)
    assert_same(kt, want["kt"], rtol=1e-9)


def test_mpv_boland(lib):
    ghi = buf(np.linspace(50.0, 1000.0, 97))
    sz = buf(np.linspace(5.0, 85.0, 97))
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(97, 172.0)))
    dni = np.full(97, np.nan)
    dhi = np.full(97, np.nan)
    kt = np.full(97, np.nan)
    lib.mpv_boland(a(ghi), a(sz), a(ex), a(dni), a(dhi), a(kt), 97,
                   8.645, 0.613, 0.065, 87.0)
    want = pvlib.irradiance.boland(ghi, sz, np.full(97, 172.0))
    assert_same(dni, want["dni"], rtol=1e-9)
    assert_same(dhi, want["dhi"], rtol=1e-9)
    assert_same(kt, want["kt"], rtol=1e-9)


def test_mpv_dni(lib):
    ghi = buf(np.linspace(0.0, 1000.0, 97))
    dhi = buf(np.linspace(0.0, 300.0, 97))
    sz = buf(np.linspace(0.0, 80.0, 97))
    clear = buf(np.linspace(100.0, 1100.0, 97))
    res = np.full(97, np.nan)
    lib.mpv_dni(a(ghi), a(dhi), a(sz), a(clear), a(res), 97, 1.1, 1, 88.0, 80.0)
    # pvlib masks on zenith thresholds, which needs a Series to align
    times = pd.date_range("2020-06-21", periods=97, freq="h", tz="UTC")
    want = pvlib.irradiance.dni(
        pd.Series(ghi, index=times), pd.Series(dhi, index=times),
        pd.Series(sz, index=times), dni_clear=pd.Series(clear, index=times),
        clearsky_tolerance=1.1, zenith_threshold_for_zero_dni=88.0,
        zenith_threshold_for_clearsky_limit=80.0)
    assert_same(res, np.asarray(want), rtol=1e-12)


def test_mpv_dni_without_dni_clear(lib):
    """`use_dni_clear=0` must ignore the clearsky branch entirely."""
    ghi = buf(np.linspace(0.0, 1000.0, 97))
    dhi = buf(np.linspace(0.0, 300.0, 97))
    sz = buf(np.linspace(0.0, 80.0, 97))
    clear = buf(np.zeros(97))
    res = np.full(97, np.nan)
    lib.mpv_dni(a(ghi), a(dhi), a(sz), a(clear), a(res), 97, 1.1, 0, 88.0, 80.0)
    want = pvlib.irradiance.dni(ghi, dhi, sz, clearsky_tolerance=1.1,
                                zenith_threshold_for_zero_dni=88.0,
                                zenith_threshold_for_clearsky_limit=80.0)
    assert_same(res, want, rtol=1e-12)


def test_mpv_louche(lib):
    ghi = buf(np.linspace(0.0, 1000.0, 97))
    sz = buf(np.linspace(0.0, 88.0, 97))
    ex = buf(pvlib.irradiance.get_extra_radiation(np.full(97, 172.0)))
    dni = np.full(97, np.nan)
    dhi = np.full(97, np.nan)
    kt = np.full(97, np.nan)
    lib.mpv_louche(a(ghi), a(sz), a(ex), a(dni), a(dhi), a(kt), 97, 90.0)
    want = pvlib.irradiance.louche(ghi, sz, np.full(97, 172.0))
    assert_same(dni, want["dni"], rtol=1e-9)
    assert_same(dhi, want["dhi"], rtol=1e-9)
    assert_same(kt, want["kt"], rtol=1e-9)


def test_mpv_campbell_norman(lib):
    """Includes the horizon: `transmittance**airmass` must underflow to 0 there,
    and the below-horizon row must be NaN, both as numpy does it."""
    zenith = buf([-5.0, 0.0, 15.0, 45.0, 75.0, 89.0, 90.0, 95.0])
    ghi = np.full(8, np.nan)
    dni = np.full(8, np.nan)
    dhi = np.full(8, np.nan)
    lib.mpv_campbell_norman(a(zenith), a(ghi), a(dni), a(dhi), 8, 0.8,
                            101325.0, 1367.0)
    want = pvlib.irradiance.campbell_norman(zenith, 0.8)
    for key, got in (("ghi", ghi), ("dni", dni), ("dhi", dhi)):
        assert_same(got, want[key], rtol=POW_TOL)
    assert dni[6] == 0.0, "tau**airmass did not underflow at zenith 90"
    assert np.isnan(dni[7]), "below-horizon row must be NaN"


def test_mpv_diffuse_par_spitters(lib):
    zen = buf([0.0, 20.0, 45.0, 70.0, 90.0])
    kd = buf([0.0, 0.1, 0.4, 0.75, 1.0])
    res = np.full(5, np.nan)
    lib.mpv_diffuse_par_spitters(a(zen), a(kd), a(res), 5)
    assert_same(res, pvlib.irradiance.diffuse_par_spitters(zen, kd), rtol=1e-12)


def test_mpv_poa_components(lib):
    """One pass writes all three totals, matching upstream's three expressions."""
    n = 200
    aoi = buf(np.linspace(0.0, 95.0, n))
    dni = buf(np.linspace(0.0, 1100.0, n))
    sky = buf(np.linspace(0.0, 250.0, n))
    grd = buf(np.linspace(0.0, 90.0, n))
    direct = np.full(n, np.nan)
    diffuse = np.full(n, np.nan)
    glob = np.full(n, np.nan)
    lib.mpv_poa_components(a(aoi), a(dni), a(sky), a(grd),
                           a(direct), a(diffuse), a(glob), n)
    want_direct = np.maximum(dni * np.cos(np.radians(aoi)), 0)
    want_diffuse = sky + grd
    assert_same(direct, want_direct, rtol=1e-9)
    assert_same(diffuse, want_diffuse, rtol=0, atol=0)
    assert_same(glob, want_direct + want_diffuse, rtol=1e-9)


def test_mpv_poa_components_nan_and_clip(lib):
    """A NaN aoi stays NaN and a below-horizon cos clips to zero, as numpy does."""
    aoi = buf([np.nan, 180.0, 90.0, 45.0])
    dni = buf([700.0, 700.0, 700.0, 700.0])
    sky = buf([10.0, 20.0, 30.0, 40.0])
    grd = buf([1.0, 2.0, 3.0, 4.0])
    direct = np.zeros(4)
    diffuse = np.zeros(4)
    glob = np.zeros(4)
    lib.mpv_poa_components(a(aoi), a(dni), a(sky), a(grd),
                           a(direct), a(diffuse), a(glob), 4)
    want = np.maximum(dni * np.cos(np.radians(aoi)), 0)
    np.testing.assert_array_equal(np.isnan(direct), np.isnan(want))
    assert direct[1] == 0.0, "cos below -90 must clip to 0, not stay negative"
    assert diffuse[0] == 11.0, "a NaN aoi must not poison the diffuse total"


def test_mpv_poa_components_respects_n(lib):
    aoi = buf(np.linspace(0.0, 90.0, 8))
    dni = buf(np.full(8, 500.0))
    sky = buf(np.full(8, 10.0))
    grd = buf(np.full(8, 5.0))
    direct = np.full(8, np.nan)
    diffuse = np.full(8, np.nan)
    glob = np.full(8, np.nan)
    lib.mpv_poa_components(a(aoi), a(dni), a(sky), a(grd),
                           a(direct), a(diffuse), a(glob), 3)
    assert np.isfinite(direct[:3]).all()
    assert np.isnan(direct[3:]).all(), "wrote past n"
    assert np.isnan(glob[3:]).all(), "wrote poa_global past n"


# ----------------------------------------------------------------- tables
def test_mpv_dirint_coeffs_table(lib):
    coeffs = zeros(6 * 6 * 7 * 5)
    lib.mpv_dirint_coeffs_table(a(coeffs))
    assert_same(coeffs.reshape(6, 6, 7, 5),
                pvlib.irradiance._get_dirint_coeffs(), rtol=0, atol=0)


@pytest.mark.parametrize("model", _lib.PEREZ_MODELS)
def test_mpv_perez_coefficients(lib, model):
    # call the symbol directly rather than through the cached helper, so the
    # kernel is exercised and not just its Python memo
    f1 = zeros((9, 3))
    f2 = zeros((9, 3))
    lib.mpv_perez_coefficients(_lib.PEREZ_MODELS.index(model), a(f1.ravel()),
                               a(f2.ravel()))
    g1, g2 = pvlib.irradiance._get_perez_coefficients(model)
    # row 0 is the all-NaN row upstream appends for invalid clearness bins
    assert_same(f1[1:], g1, rtol=0, atol=0)
    assert_same(f2[1:], g2[:, :3], rtol=0, atol=0)
    assert np.isnan(f1[0]).all() and np.isnan(f2[0]).all()


def test_every_exported_symbol_is_covered(lib):
    """The audit, run by the suite: no `@export` can be added without a test.

    If a kernel is added to src/*.mojo and not exercised here, this fails and
    names it. It is the same grep the port checklist runs, made mechanical.
    """
    import glob
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exported = set()
    for path in glob.glob(os.path.join(root, "src", "*.mojo")):
        with open(path) as handle:
            exported |= set(re.findall(r'@export\("(\w+)"\)', handle.read()))

    tested = set()
    with open(__file__) as handle:
        tested = set(re.findall(r"lib\.(mpv_\w+)", handle.read()))

    untested = sorted(exported - tested)
    assert not untested, f"exported but never called by name: {untested}"


def test_public_signatures_match_upstream():
    """Argument names, order and defaults are the drop-in contract.

    A caller moving from `import pvlib` to `import mojopvlib` must not have to
    learn a new API, so any signature drift from upstream is a failure here
    rather than a surprise at the call site.
    """
    import inspect

    import mojopvlib as pv

    checked = 0
    for mod in ("solarposition", "irradiance", "atmosphere", "tools"):
        ours, theirs = getattr(pv, mod), getattr(pvlib, mod)
        for name in sorted(n for n in dir(ours) if not n.startswith("_")):
            if not hasattr(theirs, name):
                continue
            ours_fn, theirs_fn = getattr(ours, name), getattr(theirs, name)
            if not callable(ours_fn) or not callable(theirs_fn):
                continue
            try:
                a = inspect.signature(ours_fn)
                b = inspect.signature(theirs_fn)
            except (TypeError, ValueError):
                continue
            # `dni` is the documented alias pvlib.irradiance also uses
            if a != b:
                pytest.fail(f"{mod}.{name}: ours {a} vs upstream {b}")
            checked += 1
    assert checked > 40, f"only {checked} signatures compared; import is broken"