"""Parity for the rest of `pvlib.atmosphere` against the real pvlib.

Most of these build on `**` or on `exp`/`log`, where LLVM's `pow` agrees with
libm's to ~1e-10 relative rather than exactly, so 1e-9 is plenty for the plain
arithmetic ones. The Angstrom pair subtracts two exponentials of nearly equal
magnitude, which cancels the leading digits and leaves a relative error of a
few ULP of the *inputs*; those two use `POW_TOL`, the same tolerance the main
parity file gives the power-law chains.
"""

import numpy as np
import pandas as pd
import pytest

pvlib_atmosphere = pytest.importorskip("pvlib.atmosphere")

from mojopvlib import atmosphere

RTOL = 1e-9
POW_TOL = 1e-6


def same(a, b, rtol=RTOL):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True)


def temps(n=37):
    rng = np.random.default_rng(1)
    return np.linspace(-40.0, 55.0, n) + rng.normal(0, 0.1, n)


def rhs(n=37):
    rng = np.random.default_rng(2)
    return np.linspace(1.0, 100.0, n) + rng.normal(0, 0.5, n)


def index(n=37):
    return pd.date_range("2020-03-01", periods=n, freq="h", tz="UTC")


def test_rh_from_tdew_array():
    ta, td = temps(), temps() - 8.0
    same(
        atmosphere.rh_from_tdew(ta, td),
        pvlib_atmosphere.rh_from_tdew(ta, td),
    )


def test_rh_from_tdew_scalar():
    same(
        atmosphere.rh_from_tdew(21.5, 12.25),
        pvlib_atmosphere.rh_from_tdew(21.5, 12.25),
    )


def test_rh_from_tdew_series():
    ta = pd.Series(temps(), index=index())
    td = pd.Series(temps() - 8.0, index=index())
    got = atmosphere.rh_from_tdew(ta, td)
    assert isinstance(got, pd.Series)
    pd.testing.assert_index_equal(got.index, ta.index)
    same(got, pvlib_atmosphere.rh_from_tdew(ta, td))


def test_rh_from_tdew_coeff():
    ta, td = temps(), temps() - 8.0
    coeff = (6.1078, 17.27, 237.3)
    same(
        atmosphere.rh_from_tdew(ta, td, coeff=coeff),
        pvlib_atmosphere.rh_from_tdew(ta, td, coeff=coeff),
    )


def test_tdew_from_rh_array():
    ta, rh = temps(), rhs()
    same(
        atmosphere.tdew_from_rh(ta, rh),
        pvlib_atmosphere.tdew_from_rh(ta, rh),
    )


def test_tdew_from_rh_scalar():
    same(
        atmosphere.tdew_from_rh(21.5, 47.0),
        pvlib_atmosphere.tdew_from_rh(21.5, 47.0),
    )


def test_tdew_from_rh_series():
    ta = pd.Series(temps(), index=index())
    rh = pd.Series(rhs(), index=index())
    got = atmosphere.tdew_from_rh(ta, rh)
    assert isinstance(got, pd.Series)
    pd.testing.assert_index_equal(got.index, ta.index)
    same(got, pvlib_atmosphere.tdew_from_rh(ta, rh))


def test_tdew_from_rh_coeff():
    ta, rh = temps(), rhs()
    coeff = (6.1078, 17.27, 237.3)
    same(
        atmosphere.tdew_from_rh(ta, rh, coeff=coeff),
        pvlib_atmosphere.tdew_from_rh(ta, rh, coeff=coeff),
    )


def test_angstrom_aod_at_lambda_array():
    aod0 = np.linspace(0.02, 0.5, 37)
    lambda0 = np.linspace(380.0, 800.0, 37)
    same(
        atmosphere.angstrom_aod_at_lambda(aod0, lambda0),
        pvlib_atmosphere.angstrom_aod_at_lambda(aod0, lambda0),
    )


def test_angstrom_aod_at_lambda_scalar():
    # chains exp/log; a few ULP of libm difference is expected, not 1e-9.
    same(
        atmosphere.angstrom_aod_at_lambda(0.1, 500.0),
        pvlib_atmosphere.angstrom_aod_at_lambda(0.1, 500.0),
        rtol=POW_TOL,
    )


def test_angstrom_aod_at_lambda_series():
    aod0 = pd.Series(np.linspace(0.02, 0.5, 37), index=index())
    lambda0 = pd.Series(np.linspace(380.0, 800.0, 37), index=index())
    got = atmosphere.angstrom_aod_at_lambda(aod0, lambda0)
    assert isinstance(got, pd.Series)
    pd.testing.assert_index_equal(got.index, aod0.index)
    same(got, pvlib_atmosphere.angstrom_aod_at_lambda(aod0, lambda0))


def test_angstrom_aod_at_lambda_alpha():
    aod0 = np.linspace(0.02, 0.5, 37)
    lambda0 = np.linspace(380.0, 800.0, 37)
    same(
        atmosphere.angstrom_aod_at_lambda(aod0, lambda0, alpha=0.9, lambda1=880.0),
        pvlib_atmosphere.angstrom_aod_at_lambda(aod0, lambda0, alpha=0.9, lambda1=880.0),
    )


def test_angstrom_alpha_array():
    aod1 = np.linspace(0.02, 0.5, 37)
    lambda1 = np.linspace(380.0, 700.0, 37)
    aod2 = np.linspace(0.01, 0.3, 37)
    lambda2 = np.linspace(700.0, 1000.0, 37)
    # chains exp/log over both wavelengths; a few ULP of libm difference is
    # expected, so this needs the transcendental tolerance rather than 1e-9.
    same(
        atmosphere.angstrom_alpha(aod1, lambda1, aod2, lambda2),
        pvlib_atmosphere.angstrom_alpha(aod1, lambda1, aod2, lambda2),
        rtol=POW_TOL,
    )


def test_angstrom_alpha_scalar():
    same(
        atmosphere.angstrom_alpha(0.1, 500.0, 0.2, 700.0),
        pvlib_atmosphere.angstrom_alpha(0.1, 500.0, 0.2, 700.0),
    )


def test_angstrom_alpha_series():
    aod1 = pd.Series(np.linspace(0.02, 0.5, 37), index=index())
    lambda1 = pd.Series(np.linspace(380.0, 700.0, 37), index=index())
    aod2 = pd.Series(np.linspace(0.01, 0.3, 37), index=index())
    lambda2 = pd.Series(np.linspace(700.0, 1000.0, 37), index=index())
    got = atmosphere.angstrom_alpha(aod1, lambda1, aod2, lambda2)
    assert isinstance(got, pd.Series)
    pd.testing.assert_index_equal(got.index, aod1.index)
    same(
        got,
        pvlib_atmosphere.angstrom_alpha(aod1, lambda1, aod2, lambda2),
        rtol=POW_TOL,
    )


def test_windspeed_powerlaw_array():
    ws = np.linspace(0.5, 15.0, 37)
    same(
        atmosphere.windspeed_powerlaw(ws, 10.0, 40.0, exponent=0.14),
        pvlib_atmosphere.windspeed_powerlaw(ws, 10.0, 40.0, exponent=0.14),
    )


def test_windspeed_powerlaw_scalar():
    same(
        atmosphere.windspeed_powerlaw(3.5, 9.144, 30.0, exponent=0.219),
        pvlib_atmosphere.windspeed_powerlaw(3.5, 9.144, 30.0, exponent=0.219),
    )


def test_windspeed_powerlaw_series():
    ws = pd.Series(np.linspace(0.5, 15.0, 37), index=index())
    got = atmosphere.windspeed_powerlaw(ws, 10.0, 40.0, exponent=0.14)
    assert isinstance(got, pd.Series)
    pd.testing.assert_index_equal(got.index, ws.index)
    same(got, pvlib_atmosphere.windspeed_powerlaw(ws, 10.0, 40.0, exponent=0.14))


@pytest.mark.parametrize("surface_type", [
    "unstable_air_above_open_water_surface",
    "stable_air_above_flat_open_coast",
    "neutral_air_above_human_inhabited_areas",
])
def test_windspeed_powerlaw_surface_type(surface_type):
    ws = np.linspace(0.5, 15.0, 37)
    same(
        atmosphere.windspeed_powerlaw(ws, 10.0, 40.0, surface_type=surface_type),
        pvlib_atmosphere.windspeed_powerlaw(
            ws, 10.0, 40.0, surface_type=surface_type),
    )


def test_windspeed_powerlaw_negative_is_nan():
    ws = np.array([1.0, -1.0, 4.0, -3.5])
    same(
        atmosphere.windspeed_powerlaw(ws, 10.0, 40.0, exponent=0.14),
        pvlib_atmosphere.windspeed_powerlaw(ws, 10.0, 40.0, exponent=0.14),
    )


def test_windspeed_powerlaw_negative_heights_are_nan():
    # A negative height ratio raised to a fractional exponent is complex;
    # upstream maps that to NaN via np.where. The array form is where upstream
    # actually reaches the NaN branch (its scalar form raises TypeError, a
    # documented divergence -- see the README coverage section).
    ws = np.array([2.0, 3.0, 4.0])
    heights = np.array([-10.0, 10.0, 0.0])
    same(
        atmosphere.windspeed_powerlaw(ws, heights, 40.0, exponent=0.14),
        pvlib_atmosphere.windspeed_powerlaw(ws, heights, 40.0, exponent=0.14),
    )


def test_windspeed_powerlaw_negative_scalar_height_is_nan():
    # Documented divergence: upstream raises TypeError on the scalar path
    # because numpy builds a complex before the np.where NaN guard runs. The
    # port produces the NaN the docstring and the array path describe.
    assert np.isnan(atmosphere.windspeed_powerlaw(2.0, -10.0, 40.0, exponent=0.14))
    with pytest.raises(TypeError):
        pvlib_atmosphere.windspeed_powerlaw(2.0, -10.0, 40.0, exponent=0.14)


def test_windspeed_powerlaw_neither_raises():
    with pytest.raises(ValueError) as mine:
        atmosphere.windspeed_powerlaw(2.0, 10.0, 40.0)
    with pytest.raises(ValueError) as theirs:
        pvlib_atmosphere.windspeed_powerlaw(2.0, 10.0, 40.0)
    assert str(mine.value) == str(theirs.value)


def test_windspeed_powerlaw_both_raises():
    with pytest.raises(ValueError) as mine:
        atmosphere.windspeed_powerlaw(2.0, 10.0, 40.0, exponent=0.14,
                                      surface_type="neutral_air_above_open_water_surface")
    with pytest.raises(ValueError) as theirs:
        pvlib_atmosphere.windspeed_powerlaw(2.0, 10.0, 40.0, exponent=0.14,
                                            surface_type="neutral_air_above_open_water_surface")
    assert str(mine.value) == str(theirs.value)


def test_windspeed_powerlaw_bad_surface_type_raises():
    with pytest.raises(KeyError):
        atmosphere.windspeed_powerlaw(2.0, 10.0, 40.0, surface_type="lava_field")


def test_hellmann_exponents_match_upstream():
    assert atmosphere.HELLMANN_SURFACE_EXPONENTS == \
        pvlib_atmosphere.HELLMANN_SURFACE_EXPONENTS
