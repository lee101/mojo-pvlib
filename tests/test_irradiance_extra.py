"""Parity for `campbell_norman` and `diffuse_par_spitters`.

Both models are built on `**`. LLVM's `pow` and libm's `pow` are different
implementations that agree only to ~5e-9 relative, not bit for bit, so the flat
1e-9 below is enough for the models whose exponents are small constants
(`diffuse_par_spitters`). `campbell_norman` is different: it raises a
data-driven transmittance to a data-driven airmass, and near the horizon that
airmass runs past 500, so a 5e-9 relative difference in `pow` is amplified by
the exponent into ~5e-8 of the result. That one uses `POW_TOL`.
"""

import numpy as np
import pandas as pd
import pytest

pvlib_irradiance = pytest.importorskip("pvlib.irradiance")
import pvlib.irradiance

from mojopvlib import irradiance

TOL = 1e-9
POW_TOL = 1e-6


def test_campbell_norman_array():
    zenith = np.array([-5.0, 0.0, 15.0, 45.0, 75.0, 89.0, 89.9, 95.0])
    expect = pvlib_irradiance.campbell_norman(zenith, 0.75)
    got = irradiance.campbell_norman(zenith, 0.75)
    assert list(got) == ["ghi", "dni", "dhi"]
    for key in ("ghi", "dni", "dhi"):
        np.testing.assert_allclose(got[key], expect[key], rtol=POW_TOL)


def test_campbell_norman_scalar():
    got = irradiance.campbell_norman(37.5, 0.6)
    expect = pvlib_irradiance.campbell_norman(37.5, 0.6)
    assert list(got) == ["ghi", "dni", "dhi"]
    for key in ("ghi", "dni", "dhi"):
        assert isinstance(got[key], float)
        np.testing.assert_allclose(got[key], expect[key], rtol=POW_TOL)


def test_campbell_norman_series_returns_dataframe():
    index = pd.date_range("2020-06-21", periods=4, freq="h", tz="UTC")
    zenith = pd.Series([10.0, 40.0, 80.0, 95.0], index=index)
    expect = pvlib_irradiance.campbell_norman(zenith, 0.8)
    got = irradiance.campbell_norman(zenith, 0.8)
    assert isinstance(got, pd.DataFrame)
    assert list(got.columns) == ["ghi", "dni", "dhi"]
    np.testing.assert_array_equal(got.index, index)
    for key in ("ghi", "dni", "dhi"):
        # the 95 deg element is below the horizon: airmass is NaN and the whole
        # row must be NaN on both sides, which assert_allclose does not check.
        np.testing.assert_array_equal(
            np.isnan(got[key].values), np.isnan(expect[key].values)
        )
        np.testing.assert_allclose(
            got[key].values, expect[key].values, rtol=POW_TOL, equal_nan=True
        )


def test_campbell_norman_nondefault_pressure_and_dni_extra():
    zenith = np.array([5.0, 30.0, 60.0, 88.0])
    expect = pvlib_irradiance.campbell_norman(
        zenith, 0.5, pressure=85000.0, dni_extra=1361.0
    )
    got = irradiance.campbell_norman(
        zenith, 0.5, pressure=85000.0, dni_extra=1361.0
    )
    for key in ("ghi", "dni", "dhi"):
        np.testing.assert_allclose(got[key], expect[key], rtol=POW_TOL)


def test_diffuse_par_spitters_array():
    zen = np.array([0.0, 20.0, 45.0, 70.0, 90.0])
    kd = np.array([0.0, 0.1, 0.4, 0.75, 1.0])
    expect = pvlib_irradiance.diffuse_par_spitters(zen, kd)
    np.testing.assert_allclose(
        irradiance.diffuse_par_spitters(zen, kd), expect, rtol=TOL
    )


def test_diffuse_par_spitters_scalar():
    expect = pvlib_irradiance.diffuse_par_spitters(52.0, 0.35)
    got = irradiance.diffuse_par_spitters(52.0, 0.35)
    assert isinstance(got, float)
    np.testing.assert_allclose(got, expect, rtol=TOL)


def test_diffuse_par_spitters_series():
    index = pd.date_range("2020-03-15", periods=3, freq="D", tz="UTC")
    zen = pd.Series([30.0, 60.0, 85.0], index=index)
    kd = pd.Series([0.2, 0.5, 0.9], index=index)
    expect = pvlib_irradiance.diffuse_par_spitters(zen, kd)
    got = irradiance.diffuse_par_spitters(zen, kd)
    assert isinstance(got, pd.Series)
    np.testing.assert_array_equal(got.index, index)
    np.testing.assert_allclose(got.values, expect.values, rtol=TOL)


def test_diffuse_par_spitters_fraction_endpoints():
    zen = np.array([15.0, 55.0])
    for kd in (0.0, 1.0):
        expect = pvlib_irradiance.diffuse_par_spitters(zen, kd)
        np.testing.assert_allclose(
            irradiance.diffuse_par_spitters(zen, kd), expect, rtol=TOL, atol=0.0
        )
