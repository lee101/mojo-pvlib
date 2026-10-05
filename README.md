# mojo-pvlib

[pvlib](https://github.com/pvlib/pvlib)'s solar position and irradiance
models, implemented in [Mojo](https://www.modular.com/mojo) and callable from
Python with the same API.

```python
import pandas as pd
import mojopvlib as pv

times = pd.date_range("2020-06-21", periods=24, freq="h", tz="UTC")
sp = pv.solarposition.spa_python(times, latitude=35.0, longitude=-105.0)
dni_extra = pv.irradiance.get_extra_radiation(times)

poa = pv.irradiance.get_total_irradiance(
    surface_tilt=25, surface_azimuth=180,
    solar_zenith=sp["zenith"], solar_azimuth=sp["azimuth"],
    dni=800, ghi=900, dhi=100, dni_extra=dni_extra, model="haydavies",
)
poa["poa_global"]
```

Every function keeps pvlib's name, argument order, defaults and return
convention, so moving from `import pvlib` to `import mojopvlib` is a rename.
Each one is tested against real pvlib on the same inputs and has to agree with
it; the suite is 396 tests, of which 94 live in `tests/test_kernels.py` and
name all 57 `@export`ed kernels individually, calling them through ctypes with
hand-built buffers and asserting on the values that come back, so a kernel
that is wired up and wired to the wrong buffer fails rather than shipping
green. An audit test enforces the other half: any new `@export` that no test
names by hand fails the suite. A further 20 tests cover the parallel path --
a short period run serially, the same period tiled past every size threshold
and run across the cores, and pvlib on the same data, all three of which have
to agree -- and 107 more hold the wrappers to pvlib's shape and container
rules, which is where most of the defects found in this last pass were.

## What is implemented

| module | contents |
| --- | --- |
| `solarposition` | `spa_python`, `get_solarposition`, `nrel_earthsun_distance`, `hour_angle`, `equation_of_time_spencer71`, `equation_of_time_pvcdrom`, `declination_spencer71`, `declination_cooper69`, `solar_azimuth_analytical`, `solar_zenith_analytical`, `sun_rise_set_transit_geometric` |
| `spa` | the whole NREL SPA: `solar_position_loop`, `solar_position`, `earthsun_distance`, `calculate_deltat`, `julian_day` |
| `irradiance` | `get_extra_radiation`, `aoi_projection`, `aoi`, `beam_component`, `get_ground_diffuse`, `isotropic`, `klucher`, `haydavies`, `reindl`, `king`, `perez`, `get_sky_diffuse`, `get_total_irradiance`, `poa_components`, `clearsky_index`, `clearness_index`, `clearness_index_zenith_independent`, `disc`, `dirint`, `dirindex`, `erbs`, `orgill_hollands`, `boland`, `louche`, `dni`, `campbell_norman`, `complete_irradiance`, `diffuse_par_spitters` |
| `atmosphere` | `pres2alt`, `alt2pres`, `get_relative_airmass` (all 8 models), `get_absolute_airmass`, `gueymard94_pw`, `bird_hulstrom80_aod_bb`, `kasten96_lt`, `rh_from_tdew`, `tdew_from_rh`, `angstrom_aod_at_lambda`, `angstrom_alpha`, `windspeed_powerlaw` |
| `tools` | `cosd`, `sind`, `tand`, `asind`, `acosd`, `atand` |

The Mojo side is a library in its own right: `src/spa.mojo`, `src/solarposition.mojo`,
`src/irradiance.mojo` and `src/atmosphere.mojo` are plain Mojo functions over raw
buffers, function for function and in upstream's source order, with no Python
anywhere near them. `src/spa.mojo` is the complete SPA: all 40 intermediate
functions from `julian_day` to `equation_of_time`, the nutation table walk, and
the per-timestamp loop.

### What is not covered

Nothing below is silently different; each is a deliberate omission.

- `solarposition`: `spa_c`, `pyephem`, `ephemeris`, `calc_time`,
  `pyephem_earthsun_distance`, `sun_rise_set_transit_spa` and
  `sun_rise_set_transit_ephem`. The non-NREL methods need optional
  third-party packages; the SPA sunrise solver needs the root finder in
  `pvlib.spa.transit_sunrise_sunset`.
- `irradiance`: `perez_driesse`, `erbs_driesse`, `gti_dirint`, the GTI and POA
  inversion models, `aoi_projection`'s siblings in `pvlib.tracking`, and the
  `albedo` table module (`get_ground_diffuse`'s `surface_type` lookup is
  inlined; the test suite pins the inlined copy against the whole of
  `pvlib.albedo.SURFACE_ALBEDOS`). `perez_driesse` and `erbs_driesse` need
  SciPy's spline interpolators, and `ghi_from_poa_driesse_2024` with its
  helpers `_ghi_from_poa` and `_poa_from_ghi` is built on both of them.
  `ghi_from_poa_driesse_2023` and `ghi_from_poa_driesse_2024` are the
  deprecated aliases of the same function.
- `tools`: the six degree-argument trig functions are ported; the rest of the
  module (`localize_to_utc`, `datetime_to_djd`, `djd_to_datetime`,
  `get_pandas_index`, `normalize_max2one`) is pandas and timezone bookkeeping
  rather than physics, and is left to pandas.
- `get_solarposition` supports `method='nrel_numpy'` only and raises
  `ValueError` for the others; `spa_python` accepts and ignores `how` and
  `numthreads`, which exist upstream to select a numba path.
- The Perez coefficient lookup reads a 9-row table whose row 0 is the all-NaN
  row upstream appends for invalid clearness bins; Mojo cannot index backwards,
  so the kernel adds one to the bin instead of numpy's -1. Same values, same
  NaN.
- `dirint` needs three intermediate series (DISC's airmass, kt', and
  precipitable water) that upstream holds as whole arrays between passes, so
  its kernel takes a caller-owned 3n scratch buffer. Nothing else needs one:
  `longitude_obliquity_nutation` returns its two values as a tuple rather than
  filling an `out` argument, which is also what lets `solar_position_loop`
  dispatch over its timestamps.
- `hour_angle`, `sun_rise_set_transit_geometric` and every function that takes
  a `DatetimeIndex` do their pandas work in Python, exactly as upstream does;
  only the numeric part is in Mojo.
- `spa.solar_position_loop`'s fourth parameter is named `out_` rather than
  upstream's `out`, because `out` is the name of the allocation helper this
  package uses throughout. It is the only renamed parameter in the public API,
  and `tests/test_kernels.py::test_public_signatures_match_upstream` would
  catch a second one.
- `atmosphere.windspeed_powerlaw` returns NaN for a negative height ratio on the
  scalar path, where upstream raises `TypeError`: upstream builds a complex
  number before its `np.where` NaN guard can run, and comparing complex to int
  is the error. Its own docstring and its array path both say NaN, so the port
  produces the documented value. Both behaviours are asserted side by side in
  `tests/test_atmosphere_extra.py`.
- Arguments are broadcast exactly as numpy broadcasts them, including shape:
  an n-dimensional argument comes back n-dimensional, because the kernels only
  ever see flat buffers and `_lib.shaped` puts the broadcast shape back. A
  length mismatch raises `ValueError` rather than being handed to a kernel that
  would read past the end of the shorter buffer, and the message names the
  operand shapes. A 0-d array in gives a numpy scalar out, as a ufunc does.
- The return container follows pvlib's rule for that particular expression,
  which is not always "a Series in gives a Series out". `clearness_index`
  returns a Series when any operand is one even though the first is not;
  `get_extra_radiation` unwraps a Series and keeps a `pd.Index`; the day-angle
  models return whatever their day-angle ufunc was given; `poa_components`
  builds its DataFrame from `poa_direct`, which is why a scalar `dni` against a
  Series `aoi` still comes back as a frame.
- A public function whose signature drifted from upstream's would ship silently,
  so `tests/test_kernels.py::test_public_signatures_match_upstream` compares
  every public signature in `solarposition`, `irradiance`, `atmosphere` and
  `tools` against pvlib's, argument for argument, and fails on any difference.

## Install

```
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles `src/capi.mojo` into `dist/libmojo-pvlib.so` with
`mojo build --emit shared-lib`. The build is one compilation unit and takes a
few seconds regardless of how many functions it holds. The Python package is
importable from the repo root (`PYTHONPATH=python`) or from anywhere with
`PYTHONPATH=/path/to/mojo-pvlib/python`. `import mojopvlib` builds the library
on first use if `dist/libmojo-pvlib.so` is missing or older than `src/`, so the
`build` task is a convenience rather than a prerequisite; set `MOJOPVLIB_LIB`
to point the loaders at a shared library built elsewhere.

The build needs the `max` conda package as well as `mojo`, because
`max.algorithm.parallelize` is where the multi-core path comes from; the two
are pinned to the same nightly in `pixi.toml`. `from std.algorithm import
parallelize` does not resolve on this toolchain.

## Performance

Every figure below comes from `pixi run bench` on this machine. That task holds
a machine-wide flock so a concurrently running job cannot distort it, and each
case is best-of-5 within a run. The table is the per-case median of seven
invocations:

```
Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (72 threads)
python 3.13.14, numpy 2.5.1, pvlib 0.16.1
best of 5 runs per case, median of 7 invocations of `pixi run bench`
```

This host is shared and was heavily loaded throughout, so the absolute
milliseconds are inflated relative to an idle machine and a single invocation
is not a stable number. Run to run the figures vary by a factor of two or
three; both sides of every ratio are measured in the same invocation, so the
ratios are the meaningful column, and they are the reason nothing here is
claimed to a decimal place.

| case | mojo-pvlib | pvlib | |
| --- | ---: | ---: | --- |
| `spa_python (8760 timestamps)` | 23.62 ms | 144.61 ms | **6.1x faster** |
| `nrel_earthsun_distance (8760)` | 13.48 ms | 16.80 ms | **1.2x faster** |
| `get_solarposition (8760)` | 18.84 ms | 148.12 ms | **7.9x faster** |
| `calculate_deltat (8760)` | 0.19 ms | 10.91 ms | **57.4x faster** |
| `equation_of_time_spencer71 (1M)` | 39.74 ms | 193.26 ms | **4.9x faster** |
| `get_relative_airmass (1M)` | 29.64 ms | 116.24 ms | **3.9x faster** |
| `tdew_from_rh (1M)` | 22.92 ms | 40.89 ms | **1.8x faster** |
| `campbell_norman (1M)` | 37.73 ms | 207.43 ms | **5.5x faster** |
| `diffuse_par_spitters (1M)` | 27.02 ms | 136.27 ms | **5.0x faster** |
| `tools.cosd (1M)` | 20.31 ms | 23.29 ms | **1.1x faster** |
| `get_extra_radiation (1M)` | 27.29 ms | 163.53 ms | **6.0x faster** |
| `clearsky_index (1M)` | 13.88 ms | 23.47 ms | **1.7x faster** |
| `aoi_projection (1M)` | 29.29 ms | 165.06 ms | **5.6x faster** |
| `erbs (1M)` | 86.46 ms | 426.65 ms | **4.9x faster** |
| `boland (1M)` | 93.07 ms | 327.76 ms | **3.5x faster** |
| `disc (1M)` | 45.37 ms | 485.98 ms | **10.7x faster** |
| `dni from ghi/dhi (1M)` | 26.02 ms | 33.15 ms | **1.3x faster** |
| `isotropic (1M)` | 24.31 ms | 26.29 ms | **1.1x faster** |
| `haydavies (1M)` | 41.32 ms | 329.27 ms | **8.0x faster** |
| `reindl (1M)` | 41.77 ms | 407.92 ms | **9.8x faster** |
| `perez (1M)` | 41.96 ms | 585.43 ms | **14.0x faster** |
| `dirint (8760 hourly)` | 3.25 ms | 34.66 ms | **10.7x faster** |
| `get_total_irradiance haydavies (1M)` | 187.54 ms | 580.37 ms | **3.1x faster** |

Every case is faster than pvlib on this run, from 1.1x to 57x. The shape is the
one numpy leaves on one core when it walks a Python-level loop per element,
which is what Mojo's libm calls and `parallelize` do well. The three cases with
the least margin -- `isotropic` at 1.1x, `tools.cosd` at 1.1x and `dni` at
1.3x -- are the ones with the fewest transcendentals per element, so they are
bound by loads, stores and the thread launch rather than by libm, and there is
little arithmetic to remove. Those three straddle parity run to run on a host
this loaded, so they should be read as "no reliable speedup" rather than as a
win; rerun `pixi run bench` to see the current spread.

`nrel_earthsun_distance` (1.2x here) is the smallest case with a real margin:
with `esd` set the SPA loop does a handful of operations per timestamp, so
what is left is the dispatch and the Python wrapper rather than the kernel.

### What the multi-core path is worth

`max.algorithm.parallelize`, one work item per element, one worker per physical
core. The launch is not free, and its fixed cost is what governs every kernel
here: a probe kernel that does one store per work item and nothing else costs
the same whether it is given 1000 items or 1,000,000, so the cost is a
thread-pool wake-up and join rather than anything proportional to the input.
That is why the `PAR_MIN_*` thresholds in `src/types.mojo` exist, and why
fusing two dispatches into one is worth more here than any per-element
optimisation. The four day-angle models and `poa_components` are written as one
pass for that reason, and `tests/test_parity.py` checks the fused kernels give
bit-identical answers below and above their threshold.

Three kernels deliberately stay serial at every size: `clearsky_index`,
`dirindex` and `hour_angle` have no transcendental, so their serial pass is
bound by the loads and the stores and a thread launch costs more than the work
does. `dirint`'s coefficient lookup is a table read and a multiply and stays
serial so that only one of its three passes pays for a launch.

The thresholds are re-measurable rather than asserted: raise the `PAR_MIN_*`
constants out of reach and re-run `pixi run bench` to get the serial numbers
for the same host and the same load.

### What did not pay, and why

**SIMD.** Every kernel in the hot list is libm-bound: it makes scalar `sin`,
`cos`, `exp` and `pow` calls. A probe of `std.math` over a `SIMD[float64, 4]`
loop against the same operation written scalar was run against this toolchain
during development and gained nothing on any transcendental -- `sqrt`, `exp`,
`sin`, `tan` and `**` were all within noise of scalar, and `cos` was slower on
the `SIMD` than on the scalar libm call it replaces -- so the kernels are left
scalar rather than rewritten into a form that measures the same and reads
worse. That probe is not part of `pixi run bench`, so no numbers are quoted for
it here. Two further blockers are worth naming for anyone who tries: `SIMD` has
no `.min()`/`.max()` methods in this dialect, and a `SIMD` comparison yields a
scalar `Bool`, so there is no `where`/`select` on a `SIMD` value either -- a
masked vector version of `clearsky_index` does not compile.

**The GPU.** The host API is present and working on this toolchain
(`max.gpu.host`'s `DeviceContext`, `enqueue_create_buffer`, `enqueue_copy`,
`enqueue_function`, `synchronize` all compile and run, and the device reports
itself available), but it is not shipped for a concrete reason rather than a
taste one: the NVIDIA backend in `mojo ==1.2.0.dev2026092905` has no float64
trigonometry. A kernel calling `cos` on a `Pointer[Float64]` fails the offload
pass with `constraint failed: DType.float64 is not supported for cos on NVIDIA
GPU`, and the same holds for `sin`, `tan`, `asin`, `acos` and `atan2`. `exp`,
`log`, `sqrt` and `**` do offload. Every model in this port except
`calculate_deltat` is built on the functions that do not, and supplying them
would mean writing float64 trigonometry from scratch for the device, which is
a change of numerics rather than an optimisation.

**Nested parallelism and thread reuse.** `sync_parallelize` takes no worker
count, and the two-argument `parallelize` form is documented as crashing when
called from a ctypes host, so the explicit three-argument form is used.
Handing out contiguous chunks instead of one item per element was measured
during development and made no difference, so the simpler one-item-per-element
form is kept.

## How it works

**FFI strategy.** `src/capi.mojo` is the only file with `@export`. Every
exported function takes `Int` addresses and rebuilds the pointer inside,
because `@export` rejects parametric functions and a pointer with an inferred
origin is parametric. That is the whole divergence layer: a kernel in
`src/irradiance.mojo` can be called from Mojo with real pointers, and
`capi.mojo` is the only place that knows the ABI is C. Python side,
`python/mojopvlib/_lib.py` holds one `ctypes.CDLL` and a signature per export;
the module wrappers do the numpy bookkeeping upstream's numpy calls would do
(broadcast a scalar argument to the length of the arrays, hand back a Series
when a Series went in, build a DataFrame when a `DatetimeIndex` went in).

Every buffer that crosses is float64, C-contiguous, and exactly `n` elements
long. `_lib.vectors` and `_lib.broadcast` are where that is enforced: a scalar
or length-1 argument is materialised to `n`, and any other length mismatch
raises the same `ValueError` numpy would, naming the operand shapes. That check
is the only thing standing between a caller's ragged input and a kernel reading
past the end of the shorter buffer, so it lives in one place that every wrapper
goes through. The one place a value crosses as something other than a buffer
is `campbell_norman`'s `transmittance`, `pressure` and `dni_extra`, which are
kernel scalars; `_kernel_scalar` in `python/mojopvlib/irradiance.py` accepts a
number or a length-1 array -- which is what numpy's scalar arithmetic accepts --
and refuses anything longer rather than reading its first element on the
caller's behalf.

**Memory layout.** Everything is float64 and the caller owns every buffer.
No kernel in `src/` allocates, so nothing can leak. A model is a flat
`for i in range(n)` loop over the same float64 buffers numpy would have been
handed, which is why the elementwise structure of upstream survives the port
intact. A kernel that writes more than one output writes all of them from the
same `one`, so `poa_components` reads its four inputs once instead of the three
times upstream's three numpy expressions read them. The only scratch anywhere is
DIRINT's 3n buffer, allocated by the Python caller.

**Parallelism.** Every elementwise kernel is written as a nested
`def one(i: Int)` that computes element `i` and nothing else. Below its
`PAR_MIN_*` threshold the kernel calls `one` in a serial loop; above it the
same `one` goes to `max.algorithm.parallelize` with one worker per physical
core. Elements are independent and each writes only its own slots, so the two
paths are bit-identical, and `tests/test_parity.py` checks exactly that by
running a short period below every threshold and the same period tiled past
all of them, then comparing both against pvlib. The rule that falls out of it:
a kernel that needs two intermediate arrays over the whole series gets two
dispatches, and each one costs a fixed launch, so anything that can be a single
pass over one element is written as one `one`. That is what the four day-angle
models and `poa_components` now do.

**Coefficient tables.** Mojo has no module-level mutable state, so the
constant tables upstream keeps as `np.array` globals (the 64-row L0 series,
the 63-row nutation table, the 6x6x7x5 DIRINT cube, the ten Perez coefficient
sets) are written once into a host-owned buffer by a generated
`spa_tables` / `dirint_coeffs_table` / `perez_coefficients` function in
`src/spa_tables.mojo` and `src/irradiance_tables.mojo`. The literals are
copied verbatim from pvlib 0.16.1 by `build/gen_tables.py`, which also emits
the `comptime` offsets the kernels index with, so the buffer layout and the
kernel cannot drift apart; re-running it against the installed pvlib
reproduces the checked-in files byte for byte, and
`tests/test_kernels.py` pins every table against pvlib on the way in.

**Numerics.** Two things differ from numpy, both deliberate and both measured.

The first is ordinary transcendental disagreement: LLVM's `pow` and libm's are
different implementations that agree to about 5e-9 relative rather than bit for
bit. Models built on `**` (`pres2alt`, `alt2pres`, `kasten96_lt`, the Kasten and
Gueymard airmass models) therefore differ from pvlib in the 8th or 9th
significant digit, and `pres2alt`'s subtraction of two nearly equal numbers
amplifies that to 7e-8 relative -- measured, which is why that one test and its
neighbours use a 1e-6 tolerance while the rest of the port is held to 1e-12.
Every other ported function agrees with pvlib to within a few ulp: the SPA to
4e-10 degrees over a full year of timestamps, the decomposition models to 1e-13
relative.

The second is a real defect in LLVM's `pow`, patched in `types.pow_`. For a base
in (0, 1) with a large positive exponent, where the true result underflows below
the smallest double, it returns 1.0 rather than 0.0; and for a NaN exponent it
returns +/-inf where numpy returns NaN. Both are reachable here, not
hypothetically: relative airmass goes to ~1.6e16 at zenith 90 deg, so
`campbell_norman`'s `transmittance ** airmass` has to underflow to 0 or DNI comes
back at the full extraterrestrial value, and the airmass is NaN above the
horizon, so every below-horizon zenith hits the NaN case. `pow_` detects exactly
those two cases -- the first by the defect's own signature, a base below 1
returning exactly 1.0 -- and returns what numpy returns. It is the only place in
the port where the arithmetic is not literally upstream's expression, and
`tests/test_kernels.py::test_mpv_campbell_norman` asserts both branches directly.

The test tolerances encode these two classes and nothing else is loosened: 1e-12
for straight arithmetic, 1e-9 for a single transcendental, 1e-6 for a chain that
exponentiates (where a 5e-9 relative difference in `pow` is amplified by the
exponent).

## Layout

```
src/            Mojo kernels, one compilation unit (src/capi.mojo)
build/build.sh  mojo build --emit shared-lib -> dist/libmojo-pvlib.so
build/gen_tables.py  regenerates the coefficient tables from installed pvlib
python/mojopvlib/    ctypes wrapper + the pvlib-shaped Python API
tests/          pytest: API-level parity against pvlib, shapes and containers,
                and one test per kernel
bench/bench.py  benchmarks against pvlib, prints a markdown table
```

`tests/` is five files and they answer different questions. `test_parity.py`
drives the models through the public Python API and compares against pvlib,
which is how a caller reaches them. `test_atmosphere_extra.py` and
`test_irradiance_extra.py` do the same for the functions that need their own
tolerance class. `test_shapes.py` holds the wrappers to pvlib's shape and
container rules on the same inputs, which is a different failure mode from a
value difference and was where this round's defects were. `test_kernels.py`
calls all 57 `@export`ed symbols by name through ctypes with hand-built
buffers, because reaching a kernel through the API is good coverage but weak
evidence -- a kernel wired to the wrong buffer would still ship green. It ends
with two audits that run as tests: every `@export` must be named in that file,
and every public signature must match upstream's.

## Licence

MIT, as pvlib is. See `LICENSE`.