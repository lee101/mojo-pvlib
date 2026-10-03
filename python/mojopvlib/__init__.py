"""mojo-pvlib: pvlib's solar position and irradiance models, in Mojo.

The public API mirrors `pvlib`: same module names, same function names, same
argument order and defaults.

    import mojopvlib as pv
    sp = pv.solarposition.spa_python(times, lat, lon)
    dni_extra = pv.irradiance.get_extra_radiation(times)
"""

from . import atmosphere, irradiance, solarposition, spa, tools
from ._lib import lib

__version__ = "0.1.0"

__all__ = ["atmosphere", "irradiance", "solarposition", "spa", "tools", "lib"]
