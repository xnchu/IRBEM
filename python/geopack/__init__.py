"""
geopack – Python wrapper for Geopack-2008 and Tsyganenko field models.

Wraps the compiled IRBEM shared library (libirbem.so) via ctypes.
The library is loaded automatically at import time if libirbem.so can be
found in the IRBEM source tree; call :func:`init` explicitly otherwise.

Quick start
-----------
>>> import numpy as np
>>> import geopack
>>> t = np.datetime64('2000-01-01T00:00:00')
>>> geopack.recalc(t)
>>> geopack.igrf_gsw(4.0, 0.0, 0.0)
>>> geopack.t89(3, 4.0, 0.0, 0.0)
"""

from .geopack import (
    init,
    recalc,
    igrf_geo,
    igrf_gsw,
    dip,
    sphcar,
    bspcar,
    bcarsp,
    conv_coord,
    t89,
    t96,
    t01,
    t01s,
    ts04,
    ts07,
    t96_mgnp,
    shuetal_mgnp,
    trace,
    trace_field_line,
)

__all__ = [
    'init',
    'recalc',
    'igrf_geo',
    'igrf_gsw',
    'dip',
    'sphcar',
    'bspcar',
    'bcarsp',
    'conv_coord',
    't89',
    't96',
    't01',
    't01s',
    'ts04',
    'ts07',
    't96_mgnp',
    'shuetal_mgnp',
    'trace',
    'trace_field_line',
]
