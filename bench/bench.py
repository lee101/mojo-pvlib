"""mojo-pvlib against pvlib on the same inputs.

    pixi run bench

The pixi task holds a machine-wide flock, so run it through pixi: a
concurrently running job would otherwise distort every number below.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojopvlib as pv  # noqa: E402
import pvlib  # noqa: E402
import pvlib.atmosphere  # noqa: E402
import pvlib.irradiance  # noqa: E402
import pvlib.solarposition  # noqa: E402
import pvlib.tools  # noqa: E402

REPEAT = 5


def timeit(fn) -> float:
    best = math.inf
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def machine() -> str:
    model = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model or platform.processor()} ({os.cpu_count()} threads)"


YEAR = pd.date_range("2020-01-01 00:00", periods=8760, freq="h", tz="UTC")
BIG = 1_000_000
SOLAR_ZENITH = np.linspace(5.0, 85.0, BIG)
GHI = np.clip(1000.0 * np.sin(np.radians(90.0 - SOLAR_ZENITH)), 0.0, None)
DHI = 0.25 * GHI
DNI = np.clip((GHI - DHI) / np.cos(np.radians(SOLAR_ZENITH)), 0.0, None)
TILT = np.full(BIG, 30.0)
AZIMUTH = np.full(BIG, 180.0)
SOLAR_AZIMUTH = np.linspace(0.0, 360.0, BIG)
DOY = np.full(BIG, 172.0)
DNI_EXTRA = pv.irradiance.get_extra_radiation(DOY)
AIRMASS = pv.atmosphere.get_relative_airmass(SOLAR_ZENITH)
SKY_KT = pv.irradiance.clearness_index(GHI, SOLAR_ZENITH, DNI_EXTRA)

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("spa_python (8760 timestamps)")
def _():
    return (lambda: pv.solarposition.spa_python(YEAR, 35.0, -105.0),
            lambda: pvlib.solarposition.spa_python(YEAR, 35.0, -105.0))


@case("nrel_earthsun_distance (8760)")
def _():
    return (lambda: pv.solarposition.nrel_earthsun_distance(YEAR),
            lambda: pvlib.solarposition.nrel_earthsun_distance(YEAR))


@case("get_solarposition (8760)")
def _():
    return (lambda: pv.solarposition.get_solarposition(YEAR, 35.0, -105.0),
            lambda: pvlib.solarposition.get_solarposition(YEAR, 35.0, -105.0))


@case("calculate_deltat (8760)")
def _():
    year = np.full(8760, 2020.0)
    month = np.full(8760, 6.0)
    return (lambda: pv.spa.calculate_deltat(year, month),
            lambda: pvlib.spa.calculate_deltat(year, month))


@case("equation_of_time_spencer71 (1M)")
def _():
    doy = np.tile(np.arange(1.0, 366.0), BIG // 365)
    return (lambda: pv.solarposition.equation_of_time_spencer71(doy),
            lambda: pvlib.solarposition.equation_of_time_spencer71(doy))


@case("get_relative_airmass (1M)")
def _():
    return (lambda: pv.atmosphere.get_relative_airmass(SOLAR_ZENITH),
            lambda: pvlib.atmosphere.get_relative_airmass(SOLAR_ZENITH))



@case("tdew_from_rh (1M)")
def _():
    temp_air = np.linspace(-20.0, 40.0, BIG)
    rh = np.linspace(5.0, 100.0, BIG)
    return (lambda: pv.atmosphere.tdew_from_rh(temp_air, rh),
            lambda: pvlib.atmosphere.tdew_from_rh(temp_air, rh))


@case("campbell_norman (1M)")
def _():
    return (lambda: pv.irradiance.campbell_norman(SOLAR_ZENITH, 0.7)["ghi"],
            lambda: pvlib.irradiance.campbell_norman(SOLAR_ZENITH, 0.7)["ghi"])


@case("diffuse_par_spitters (1M)")
def _():
    zenith = np.linspace(0.0, 80.0, BIG)
    kd = np.linspace(0.01, 1.0, BIG)
    return (lambda: pv.irradiance.diffuse_par_spitters(zenith, kd),
            lambda: pvlib.irradiance.diffuse_par_spitters(zenith, kd))

@case("tools.cosd (1M)")
def _():
    return (lambda: pv.tools.cosd(SOLAR_ZENITH),
            lambda: pvlib.tools.cosd(SOLAR_ZENITH))


@case("get_extra_radiation (1M)")
def _():
    doy = np.tile(np.arange(1.0, 366.0), BIG // 365)
    return (lambda: pv.irradiance.get_extra_radiation(doy),
            lambda: pvlib.irradiance.get_extra_radiation(doy))


@case("clearsky_index (1M)")
def _():
    return (lambda: pv.irradiance.clearsky_index(GHI, GHI * 0.9 + 10.0),
            lambda: pvlib.irradiance.clearsky_index(GHI, GHI * 0.9 + 10.0))


@case("aoi_projection (1M)")
def _():
    return (lambda: pv.irradiance.aoi_projection(TILT, AZIMUTH, SOLAR_ZENITH,
                                                 SOLAR_AZIMUTH),
            lambda: pvlib.irradiance.aoi_projection(TILT, AZIMUTH, SOLAR_ZENITH,
                                                    SOLAR_AZIMUTH))


@case("erbs (1M)")
def _():
    return (lambda: pv.irradiance.erbs(GHI, SOLAR_ZENITH, DOY)["dni"],
            lambda: pvlib.irradiance.erbs(GHI, SOLAR_ZENITH, DOY)["dni"])


@case("boland (1M)")
def _():
    return (lambda: pv.irradiance.boland(GHI, SOLAR_ZENITH, DOY)["dni"],
            lambda: pvlib.irradiance.boland(GHI, SOLAR_ZENITH, DOY)["dni"])


@case("disc (1M)")
def _():
    return (lambda: pv.irradiance.disc(GHI, SOLAR_ZENITH, DOY)["dni"],
            lambda: pvlib.irradiance.disc(GHI, SOLAR_ZENITH, DOY)["dni"])


@case("dni from ghi/dhi (1M)")
def _():
    return (lambda: pv.irradiance.dni(GHI, DHI, SOLAR_ZENITH),
            lambda: pvlib.irradiance.dni(GHI, DHI, SOLAR_ZENITH))


@case("isotropic (1M)")
def _():
    return (lambda: pv.irradiance.isotropic(TILT, DHI),
            lambda: pvlib.irradiance.isotropic(TILT, DHI))


@case("haydavies (1M)")
def _():
    return (lambda: pv.irradiance.haydavies(TILT, AZIMUTH, DHI, DNI, DNI_EXTRA,
                                            SOLAR_ZENITH, SOLAR_AZIMUTH),
            lambda: pvlib.irradiance.haydavies(TILT, AZIMUTH, DHI, DNI, DNI_EXTRA,
                                               SOLAR_ZENITH, SOLAR_AZIMUTH))


@case("reindl (1M)")
def _():
    args = (TILT, AZIMUTH, DHI, DNI, GHI, DNI_EXTRA, SOLAR_ZENITH, SOLAR_AZIMUTH)
    return (lambda: pv.irradiance.reindl(*args),
            lambda: pvlib.irradiance.reindl(*args))


@case("perez (1M)")
def _():
    args = (TILT, AZIMUTH, DHI, DNI, DNI_EXTRA, SOLAR_ZENITH, SOLAR_AZIMUTH,
            AIRMASS)
    return (lambda: pv.irradiance.perez(*args),
            lambda: pvlib.irradiance.perez(*args))


@case("dirint (8760 hourly)")
def _():
    return (lambda: pv.irradiance.dirint(GHI[:8760], SOLAR_ZENITH[:8760], YEAR),
            lambda: pvlib.irradiance.dirint(GHI[:8760], SOLAR_ZENITH[:8760], YEAR))


@case("get_total_irradiance haydavies (1M)")
def _():
    args = (TILT, AZIMUTH, SOLAR_ZENITH, SOLAR_AZIMUTH, DNI, GHI, DHI)
    kw = dict(dni_extra=DNI_EXTRA, airmass=AIRMASS, model="haydavies")
    return (lambda: pv.irradiance.get_total_irradiance(*args, **kw),
            lambda: pvlib.irradiance.get_total_irradiance(*args, **kw))


def main() -> int:
    print(f"# mojo-pvlib benchmark\n")
    print(f"pvlib {pvlib.__version__}, mojo-pvlib {pv.__version__}")
    print(f"machine: {machine()}")
    print(f"python {platform.python_version()}, numpy {np.__version__}")
    print(f"best of {REPEAT} runs\n")
    print("| case | mojo-pvlib | pvlib | |")
    print("| --- | ---: | ---: | --- |")
    rows = []
    for name, fn in CASES:
        ours, theirs = fn()
        t_ours = timeit(ours)
        t_theirs = timeit(theirs)
        ratio = t_theirs / t_ours
        verdict = f"**{ratio:.1f}x faster**" if ratio >= 1.05 else (
            f"{1 / ratio:.1f}x slower" if ratio < 0.95 else "about equal")
        print(f"| {name} | {t_ours * 1e3:.2f} ms | {t_theirs * 1e3:.2f} ms | {verdict} |")
        rows.append(ratio)
    print(f"\nslowest case: {min(rows):.2f}x of pvlib, "
          f"fastest: {max(rows):.2f}x of pvlib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
