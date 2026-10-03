"""Return shape and return container, compared against pvlib case by case.

The kernels are elementwise over flat float64 buffers, so an n-dimensional
argument is flattened on the way into the ABI and has to come back in the
shape numpy's broadcasting rules would have produced. These tests hold the
wrappers to that, and to the container pvlib picks -- a Series in gives a
Series out where pvlib preserves one, a pandas Index stays an Index, an
ndarray stays an ndarray -- because a silently flattened result is the kind of
defect that passes a values-only test suite.

Every case is run against pvlib on the same inputs and compared on container,
shape and values, so a divergence has to be reproduced before it is a fix.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

pvlib = pytest.importorskip("pvlib")
import pvlib.atmosphere
import pvlib.irradiance
import pvlib.solarposition
import pvlib.tools

from mojopvlib import atmosphere, irradiance, solarposition, tools

# `pres2alt` and `alt2pres` subtract nearly equal numbers built on `**`, which
# is where LLVM's pow and libm's disagree the most; everything else here is
# straight arithmetic and is held far tighter.
POW_TOL = 1e-6
TOL = 1e-11


def agree(ours, theirs, rtol=TOL, what=""):
    """Container, shape and values, all against pvlib."""
    assert type(ours) is type(theirs), (
        f"{what}: container {type(ours).__name__} vs {type(theirs).__name__}"
    )
    if isinstance(theirs, pd.Series):
        assert ours.index.equals(theirs.index), f"{what}: index"
        assert ours.name == theirs.name, f"{what}: name"
    if isinstance(theirs, (dict, pd.DataFrame)):
        assert set(ours) == set(theirs), f"{what}: keys"
        for key in theirs:
            agree(ours[key], theirs[key], rtol, f"{what}[{key}]")
        return
    a = np.asarray(ours, dtype=float)
    b = np.asarray(theirs, dtype=float)
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    np.testing.assert_allclose(a, b, rtol=rtol, equal_nan=True)


COLS = list("abcde")
TIMES = pd.date_range("2020-06-21", periods=5, freq="h", tz="UTC")


def S(v):
    return pd.Series(np.asarray(v, dtype=float), index=COLS)


def D2(v):
    """A (2, 3) block, so a 6-element answer has to come back 2-dimensional."""
    return np.full((2, 3), float(v))


# ------------------------------------------------------------------- tools
@pytest.mark.parametrize("name", ["cosd", "sind", "tand", "asind", "acosd",
                                  "atand"])
@pytest.mark.parametrize("kind", ["scalar", "one_d", "series", "zero_d",
                                  "two_d"])
def test_trig_shape_and_container(name, kind):
    value = 30.0 if name != "asind" and name != "acosd" else 0.5
    arg = {
        "scalar": value,
        "zero_d": np.array(value),
        "one_d": np.linspace(1.0, value, 5),
        "series": S(np.linspace(1.0, value, 5)),
        "two_d": D2(value),
    }[kind]
    ours = getattr(tools, name)(arg)
    theirs = getattr(pvlib.tools, name)(arg)
    agree(ours, theirs, what=f"tools.{name}/{kind}")
    if kind == "two_d":
        assert ours.shape == (2, 3), "a 2-D argument must not come back flat"


def test_scalar_in_is_a_numpy_scalar_like_a_ufunc():
    """`np.float64` is what a ufunc hands back for a scalar argument, and the
    distinction is visible to `isinstance`."""
    ours = tools.cosd(30.0)
    theirs = pvlib.tools.cosd(30.0)
    assert type(ours) is type(theirs), (type(ours), type(theirs))
    assert np.isscalar(ours)
    assert float(ours) == pytest.approx(float(theirs), abs=1e-15)


# ------------------------------------------------------------ solarposition
@pytest.mark.parametrize("name", ["equation_of_time_spencer71",
                                  "equation_of_time_pvcdrom",
                                  "declination_spencer71",
                                  "declination_cooper69"])
def test_day_angle_models_unwrap_a_series(name):
    """`dayofyear` is documented as an array, and pvlib's ufunc unwraps a
    Series to its values. The old `result()` treated any index-shaped object
    as a Series and handed back a Series carrying day numbers as labels."""
    ours = getattr(solarposition, name)(S([10.0, 80.0, 172.0, 355.0, 1.0]))
    theirs = getattr(pvlib.solarposition, name)(
        S([10.0, 80.0, 172.0, 355.0, 1.0]))
    agree(ours, theirs, what=f"{name}/Series")


def test_solar_zenith_analytical_two_dimensional():
    lat, ha, dec = D2(0.5), D2(0.1), D2(0.2)
    ours = solarposition.solar_zenith_analytical(lat, ha, dec)
    agree(ours, pvlib.solarposition.solar_zenith_analytical(lat, ha, dec),
          what="solar_zenith_analytical")
    zen = np.full((2, 3), 0.4)
    agree(solarposition.solar_azimuth_analytical(lat, ha, dec, zen),
          pvlib.solarposition.solar_azimuth_analytical(lat, ha, dec, zen),
          what="solar_azimuth_analytical")


@pytest.mark.parametrize("series_at", ["latitude", "hourangle", "declination"])
def test_solar_azimuth_analytical_container_follows_the_last_ufunc(series_at):
    """`zenith` is an *input* here and upstream builds `numer`/`denom` with
    plain `np.cos`/`np.sin`, which unwrap a Series to its values, so the
    container is decided by `np.sign(hourangle)` -- the last ufunc in the
    expression -- and `zenith` takes no part in it at all."""
    args = {
        "latitude": (S(np.full(5, 0.5)), np.full(5, 0.1), np.full(5, 0.2)),
        "hourangle": (np.full(5, 0.5), S(np.full(5, 0.1)), np.full(5, 0.2)),
        "declination": (np.full(5, 0.5), np.full(5, 0.1), S(np.full(5, 0.2))),
    }[series_at]
    agree(solarposition.solar_azimuth_analytical(*args, np.full(5, 0.4)),
          pvlib.solarposition.solar_azimuth_analytical(*args, np.full(5, 0.4)),
          what=f"solar_azimuth_analytical/Series {series_at}")


# --------------------------------------------------------------- atmosphere
@pytest.mark.parametrize("name,args", [
    ("pres2alt", (D2(95000.0),)),
    ("alt2pres", (D2(1500.0),)),
    ("get_absolute_airmass", (D2(1.5),)),
    ("get_relative_airmass", (D2(45.0),)),
    ("bird_hulstrom80_aod_bb", (D2(0.1), D2(0.2))),
    ("kasten96_lt", (D2(1.5), D2(1.0), D2(0.1))),
    ("gueymard94_pw", (D2(20.0), D2(50.0))),
    ("rh_from_tdew", (D2(20.0), D2(10.0))),
    ("tdew_from_rh", (D2(20.0), D2(50.0))),
    ("angstrom_alpha", (D2(0.1), 500.0, D2(0.2), 700.0)),
    ("angstrom_aod_at_lambda", (D2(0.1), D2(500.0), D2(0.2))),
    ("windspeed_powerlaw", (D2(5.0), 10.0, D2(2.0), 0.14)),
])
def test_atmosphere_two_dimensional(name, args):
    agree(getattr(atmosphere, name)(*args),
          getattr(pvlib.atmosphere, name)(*args), POW_TOL, f"atmosphere.{name}")


# --------------------------------------------------------------- irradiance
@pytest.mark.parametrize("kind,value", [
    ("Index", pd.Index([10.0, 80.0, 172.0])),
    ("Timestamp", pd.Timestamp("2020-06-21")),
    ("date", datetime.date(2020, 6, 21)),
    ("datetime", datetime.datetime(2020, 6, 21, 12)),
    ("numpy datetime64", np.datetime64("2020-06-21")),
    ("int", 172),
    ("float", 172.0),
    ("doy array", np.array([10.0, 80.0, 172.0])),
    ("Series of doy", S([10.0, 80.0, 172.0, 355.0, 1.0])),
    # `nrel` needs a 1-D day-of-year series on both sides (it is turned into a
    # DatetimeIndex), so the 2-D case is not one either implementation can
    # answer; it is listed here so that is explicit rather than accidental
    ("2-D doy, spencer/asce only", D2(172.0)),
    ("DatetimeIndex", TIMES[:3]),
])
def test_get_extra_radiation_output_container(kind, value):
    """Upstream picks the output container from the input type before doing any
    arithmetic, and it applies to `asce` and `spencer` just as much as to
    `nrel`: a `DatetimeIndex` gives a Series, a scalar datetime gives a bare
    number, and everything else gives a bare ndarray with any Series unwrapped.
    The old port only reproduced part of that, so a `Timestamp` came back as a
    one-element ndarray and a `pd.Index` as a plain array."""
    methods = ("spencer", "asce") if kind.startswith("2-D") else (
        "spencer", "asce", "nrel")
    for method in methods:
        try:
            theirs = pvlib.irradiance.get_extra_radiation(value, method=method)
        except Exception:
            with pytest.raises(Exception):
                irradiance.get_extra_radiation(value, method=method)
            continue
        ours = irradiance.get_extra_radiation(value, method=method)
        if isinstance(theirs, float):
            # a scalar day of year goes through the scalar arithmetic
            # upstream; a 1-D answer would be a different type entirely
            assert np.ndim(ours) == 0, (
                f"{kind}/{method}: scalar in, {type(ours).__name__} out")
            assert float(ours) == pytest.approx(theirs, rel=TOL)
            continue
        agree(ours, theirs, what=f"get_extra_radiation/{kind}/{method}")


def test_get_extra_radiation_two_dimensional():
    """The two closed-form methods go straight through the day-angle ufunc, so
    a 2-D input keeps its 2-D shape."""
    value = D2(172.0)
    for method in ("spencer", "asce"):
        ours = irradiance.get_extra_radiation(value, method=method)
        assert ours.shape == (2, 3)
        agree(ours, pvlib.irradiance.get_extra_radiation(value, method=method),
              what=f"get_extra_radiation 2d/{method}")


def test_get_extra_radiation_deltat_drivers_need_a_one_dimensional_index():
    """`erbs` and friends drive the SPA from a `DatetimeIndex`, which upstream
    builds from the flat day-of-year array, so a 2-D day-of-year input is not a
    case either side can answer. What matters here is that a 1-D input still
    works and that the day of year is measured from the right day."""
    value = np.full(5, 172.0)
    for name in ("erbs", "orgill_hollands", "boland", "disc", "louche"):
        ours = getattr(irradiance, name)(np.full(5, 500.0), np.full(5, 45.0),
                                        value)
        theirs = getattr(pvlib.irradiance, name)(np.full(5, 500.0),
                                                 np.full(5, 45.0), value)
        agree(ours, theirs, POW_TOL, f"{name} from day-of-year array")


def test_clearness_index_follows_the_series_not_the_first_argument():
    """Upstream returns a Series here because a middle argument is one. The
    old `result()` walked the arguments in order and stopped at the first
    Series-like, so a plain ndarray in front of it changed the container."""
    ours = irradiance.clearness_index(np.full(5, 500.0), S(np.full(5, 45.0)),
                                      np.full(5, 1367.0))
    theirs = pvlib.irradiance.clearness_index(np.full(5, 500.0),
                                              S(np.full(5, 45.0)),
                                              np.full(5, 1367.0))
    agree(ours, theirs, what="clearness_index")


def test_clearness_index_two_dimensional():
    agree(irradiance.clearness_index(D2(500.0), D2(45.0), D2(1367.0)),
          pvlib.irradiance.clearness_index(D2(500.0), D2(45.0), D2(1367.0)),
          what="clearness_index 2d")


def test_isotropic_components_follow_the_sky_value():
    """`return_components=True` wraps the dict in a DataFrame only when
    `poa_sky_diffuse` is a Series, which is decided by the operands that
    actually multiply into it -- not by whichever argument happens to be one."""
    agree(irradiance.isotropic(S(np.full(5, 30.0)), S(np.full(5, 100.0)),
                               return_components=True),
          pvlib.irradiance.isotropic(S(np.full(5, 30.0)), S(np.full(5, 100.0)),
                                     return_components=True),
          what="isotropic components/all Series")
    agree(irradiance.isotropic(np.full(5, 30.0), S(np.full(5, 100.0)),
                               return_components=True),
          pvlib.irradiance.isotropic(np.full(5, 30.0), S(np.full(5, 100.0)),
                                     return_components=True),
          what="isotropic components/Series dhi")


@pytest.mark.parametrize("name,args", [
    ("isotropic", (D2(30.0), D2(100.0))),
    ("king", (D2(30.0), D2(100.0), D2(900.0), D2(45.0))),
    ("klucher", (D2(30.0), 180.0, D2(100.0), D2(900.0), D2(45.0), 180.0)),
    ("haydavies", (D2(30.0), 180.0, D2(100.0), D2(800.0), D2(1366.0),
                   D2(45.0), 180.0)),
    ("reindl", (D2(30.0), 180.0, D2(100.0), D2(800.0), D2(900.0), D2(1366.0),
                D2(45.0), 180.0)),
    ("perez", (D2(30.0), 180.0, D2(100.0), D2(800.0), D2(1366.0), D2(45.0),
              180.0, D2(1.4))),
    ("aoi", (D2(30.0), 180.0, D2(45.0), 180.0)),
    ("aoi_projection", (D2(30.0), 180.0, D2(45.0), 180.0)),
    ("beam_component", (D2(30.0), 180.0, D2(45.0), 180.0, D2(800.0))),
    ("get_ground_diffuse", (D2(30.0), D2(800.0))),
    ("dni", (D2(500.0), D2(100.0), D2(45.0))),
    ("clearsky_index", (D2(500.0), D2(800.0))),
    ("diffuse_par_spitters", (D2(45.0), D2(0.3))),
])
def test_irradiance_two_dimensional(name, args):
    agree(getattr(irradiance, name)(*args),
          getattr(pvlib.irradiance, name)(*args), what=f"irradiance.{name} 2d")


@pytest.mark.parametrize("name,args", [
    ("isotropic", (D2(30.0), D2(100.0))),
    ("haydavies", (D2(30.0), 180.0, D2(100.0), D2(800.0), D2(1366.0),
                   D2(45.0), 180.0)),
    ("reindl", (D2(30.0), 180.0, D2(100.0), D2(800.0), D2(900.0), D2(1366.0),
                D2(45.0), 180.0)),
    ("perez", (D2(30.0), 180.0, D2(100.0), D2(800.0), D2(1366.0), D2(45.0),
              180.0, D2(1.4))),
])
def test_diffuse_components_two_dimensional(name, args):
    agree(getattr(irradiance, name)(*args, return_components=True),
          getattr(pvlib.irradiance, name)(*args, return_components=True),
          what=f"irradiance.{name} 2d components")


@pytest.mark.parametrize("name", ["erbs", "boland", "orgill_hollands", "disc",
                                  "louche"])
def test_decompositions_two_dimensional(name):
    kwargs = {"orgill_hollands": {
        "dni_extra": pvlib.irradiance.get_extra_radiation(D2(172.0))},
        "disc": {}, "erbs": {}, "boland": {}, "louche": {}}[name]
    agree(getattr(irradiance, name)(D2(500.0), D2(45.0), D2(172.0), **kwargs),
          getattr(pvlib.irradiance, name)(D2(500.0), D2(45.0), D2(172.0),
                                         **kwargs),
          what=f"irradiance.{name} 2d")


def test_campbell_norman_two_dimensional():
    agree(irradiance.campbell_norman(D2(45.0), 0.8),
          pvlib.irradiance.campbell_norman(D2(45.0), 0.8), POW_TOL,
          "campbell_norman 2d")


def test_campbell_norman_rejects_a_longer_scalar_coefficient():
    """`transmittance`, `pressure` and `dni_extra` are kernel scalars, so a
    longer array cannot be honoured. numpy refuses it; reading its first
    element silently would be the alternative, and a length-1 array would
    have been read out of bounds before this check existed."""
    for kwargs in ({"transmittance": D2(0.8)}, {"pressure": D2(90000.0)},
                   {"dni_extra": D2(1366.0)}):
        with pytest.raises(TypeError):
            irradiance.campbell_norman(np.full(5, 45.0), **kwargs)


def test_campbell_norman_length_one_scalar_coefficient():
    """numpy reads the first element of a length-1 operand, so the port does
    the same rather than refusing a value numpy accepts."""
    agree(irradiance.campbell_norman(np.full(5, 45.0), np.array([0.8])),
          pvlib.irradiance.campbell_norman(np.full(5, 45.0), np.array([0.8])),
          POW_TOL, "campbell_norman length-one coefficient")


def test_poa_components_two_dimensional():
    agree(irradiance.poa_components(D2(45.0), D2(800.0), D2(100.0), D2(10.0)),
          pvlib.irradiance.poa_components(D2(45.0), D2(800.0), D2(100.0),
                                          D2(10.0)),
          1e-12, "poa_components 2d")


def test_get_sky_diffuse_two_dimensional():
    args = tuple(D2(v) for v in (30.0, 180.0, 45.0, 180.0, 800.0, 900.0, 100.0,
                                 1366.0))
    agree(irradiance.get_sky_diffuse(*args),
          pvlib.irradiance.get_sky_diffuse(*args),
          what="get_sky_diffuse 2d")


@pytest.mark.parametrize("model", ["isotropic", "klucher", "haydavies",
                                   "reindl", "perez"])
def test_get_total_irradiance_two_dimensional(model):
    args = tuple(D2(v) for v in (30.0, 180.0, 45.0, 180.0, 800.0, 900.0, 100.0))
    kwargs = dict(dni_extra=D2(1366.0), model=model)
    agree(irradiance.get_total_irradiance(*args, **kwargs),
          pvlib.irradiance.get_total_irradiance(*args, **kwargs),
          what=f"get_total_irradiance 2d/{model}")


@pytest.mark.parametrize("model", ["isotropic", "haydavies", "reindl",
                                   "perez"])
def test_get_total_irradiance_two_dimensional_components(model):
    # `diffuse_components=True` is rejected for klucher and king upstream, so
    # those two are not in this list
    args = tuple(D2(v) for v in (30.0, 180.0, 45.0, 180.0, 800.0, 900.0, 100.0))
    kwargs = dict(dni_extra=D2(1366.0), model=model, diffuse_components=True)
    agree(irradiance.get_total_irradiance(*args, **kwargs),
          pvlib.irradiance.get_total_irradiance(*args, **kwargs),
          what=f"get_total_irradiance 2d/{model}/components")


def test_get_total_irradiance_container_comes_from_the_arguments():
    """Upstream wraps the totals in a DataFrame when `poa_direct` is a Series,
    which happens when `aoi` or `dni` is one -- even if the sky term is a bare
    array. Deciding it from the sky term instead silently turned a DataFrame
    into a dict of Series for a scalar-dni caller."""
    ours = irradiance.get_total_irradiance(
        S(np.full(5, 30.0)), 180.0, np.full(5, 45.0), 180.0,
        np.full(5, 800.0), np.full(5, 900.0), np.full(5, 100.0),
        dni_extra=S(np.full(5, 1366.0)), model="haydavies")
    theirs = pvlib.irradiance.get_total_irradiance(
        S(np.full(5, 30.0)), 180.0, np.full(5, 45.0), 180.0,
        np.full(5, 800.0), np.full(5, 900.0), np.full(5, 100.0),
        dni_extra=S(np.full(5, 1366.0)), model="haydavies")
    assert isinstance(theirs, pd.DataFrame), "upstream should build a frame here"
    agree(ours, theirs, what="get_total_irradiance/Series aoi")


def test_complete_irradiance_two_dimensional_matches_upstream():
    args = (D2(45.0), D2(500.0), D2(100.0), None)
    with pytest.raises(ValueError) as ours:
        irradiance.complete_irradiance(*args)
    with pytest.raises(ValueError) as theirs:
        pvlib.irradiance.complete_irradiance(*args)
    assert str(ours.value) == str(theirs.value)


def test_object_dtype_is_refused_rather_than_silently_stringified():
    """`f64` answers the C-order question on the original dtype, so an object
    array never reaches a kernel as a float64 reinterpretation of addresses."""
    for value in (np.array([172.0, 100.0], dtype=object),
                  np.array(["172", "100"], dtype=object)):
        with pytest.raises(TypeError):
            irradiance.get_extra_radiation(value)


def test_mismatched_lengths_report_the_shapes_pvlib_would():
    """Both sides have to name the operand shapes, not just refuse the call:
    the message is how a caller works out which argument is ragged."""
    with pytest.raises(ValueError) as ours:
        irradiance.aoi(np.full((2, 3), 30.0), 180.0, np.full(5, 45.0), 180.0)
    with pytest.raises(ValueError) as theirs:
        pvlib.irradiance.aoi(np.full((2, 3), 30.0), 180.0, np.full(5, 45.0), 180.0)
    assert "shapes (2, 3) () (5,) ()" in str(ours.value), str(ours.value)
    assert "shapes (2,3) (5,)" in str(theirs.value), str(theirs.value)