"""
Python ctypes wrapper for Geopack-2008 and Tsyganenko field models.

All functions operate in GSW (= GSM when solar wind is along -X) coordinates
unless noted. Lengths are in Earth radii (1 RE = 6371.2 km), fields in nT.

Usage
-----
import geopack
import numpy as np

geopack.init()  # or auto-initialized on import
t = np.datetime64('2000-01-01T00:00:00')
geopack.recalc(t)

br, bt, bp = geopack.igrf_geo(1.0, 0.5, 0.3)
x, y, z = geopack.conv_coord(4.0, 0.0, 0.0, 'GSM', 'GEO')
bx, by, bz = geopack.t89(3, -4.0, 2.0, 1.0)
"""

import ctypes
import datetime
import sys
from collections import deque
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Module-level state
# --------------------------------------------------------------------------- #
_lib = None
_TRANSFORMS = None   # populated by init()
_MODEL_FUNCS = None  # populated by init()
_GP_EXT_FUNCS = None  # geopack-interface adapters for EXNAME, populated by init()
_recalc_called = False   # set True by recalc(); checked by _check_recalc()

# ---------------------------------------------------------------------------
# ctypes function type for the 9-argument EXNAME interface expected by
# RHAND_08 inside TRACE_08:  EXNAME(IOPT, PARMOD, PS, X, Y, Z, BX, BY, BZ)
# All Fortran arguments are passed by reference (as C pointers).
# ---------------------------------------------------------------------------
_EXNAME_CFUNCTYPE = ctypes.CFUNCTYPE(
    None,                              # void return
    ctypes.POINTER(ctypes.c_int),     # IOPT*
    ctypes.POINTER(ctypes.c_double),  # PARMOD* (first element of 10-elem array)
    ctypes.POINTER(ctypes.c_double),  # PS* (dipole tilt angle, radians)
    ctypes.POINTER(ctypes.c_double),  # X*
    ctypes.POINTER(ctypes.c_double),  # Y*
    ctypes.POINTER(ctypes.c_double),  # Z*
    ctypes.POINTER(ctypes.c_double),  # BX* (output)
    ctypes.POINTER(ctypes.c_double),  # BY* (output)
    ctypes.POINTER(ctypes.c_double),  # BZ* (output)
)

_ORDERED_MAGINPUT_KEYS = [
    'Kp', 'Dst', 'dens', 'velo', 'Pdyn', 'ByIMF', 'BzIMF',
    'G1', 'G2', 'G3', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'AL',
]

# kext integer values as used by IRBEM (from extModels list index):
# None=0, MF75=1, TS87=2, TL87=3, T89=4, OPQ77=5, OPD88=6, T96=7,
# OM97=8, T01=9, T01S=10, T04=11, A00=12, T07=13, MT=14
_KEXT_MAP = {
    'none':   0,
    'dipole': 0,
    't89':    4,
    't96':    7,
    't01':    9,
    't01s':  10,
    'ts04':  11,
    'ts07':  13,
}


# --------------------------------------------------------------------------- #
# Library loading
# --------------------------------------------------------------------------- #
def init(path=None):
    """
    Load the IRBEM shared library and set up internal state.

    Called automatically at import time with a graceful fallback if the library
    is not found.  Call explicitly with ``path`` to specify the library location.

    Parameters
    ----------
    path : str or Path, optional
        Full path to ``libirbem.so`` / ``libirbem.dll``.  If *None* the module
        directory tree is searched recursively.
    """
    global _lib, _TRANSFORMS, _MODEL_FUNCS

    if path is None:
        lib_name = 'libirbem.dll' if sys.platform in ('win32', 'cygwin') else 'libirbem.so'
        search_root = Path(__file__).resolve().parent.parent.parent
        candidates = list(search_root.rglob(lib_name))
        if not candidates:
            raise FileNotFoundError(
                f"'{lib_name}' not found under {search_root}. "
                "Build the IRBEM library first (see CLAUDE.md)."
            )
        path = str(candidates[0])

    loader = ctypes.WinDLL if sys.platform in ('win32', 'cygwin') else ctypes.CDLL
    _lib = loader(str(path))

    # Coordinate transform table: (src, dst) -> (fortran_func, J_direction)
    # J=+1 transforms in the "forward" direction, J=-1 in the reverse.
    _TRANSFORMS = {
        ('GEI', 'GEO'): (_lib.geigeo_08_,  1),
        ('GEO', 'GEI'): (_lib.geigeo_08_, -1),
        ('GEO', 'MAG'): (_lib.geomag_08_,  1),
        ('MAG', 'GEO'): (_lib.geomag_08_, -1),
        ('MAG', 'SM'):  (_lib.magsm_08_,   1),
        ('SM',  'MAG'): (_lib.magsm_08_,  -1),
        ('SM',  'GSW'): (_lib.smgsw_08_,   1),
        ('GSW', 'SM'):  (_lib.smgsw_08_,  -1),
        ('GEO', 'GSW'): (_lib.geogsw_08_,  1),
        ('GSW', 'GEO'): (_lib.geogsw_08_, -1),
        ('GSW', 'GSE'): (_lib.gswgse_08_,  1),
        ('GSE', 'GSW'): (_lib.gswgse_08_, -1),
    }

    # Map model name -> function pointer (used by trace() for INNAME only).
    # INNAME calls INNAME(X, Y, Z, BX, BY, BZ) – 6 args, matching IGRF_GSW_08 / DIP_08.
    _MODEL_FUNCS = {
        'igrf':  _lib.igrf_gsw_08_,
        'dip':   _lib.dip_08_,
        't89':   _lib.t89c_,
        't96':   _lib.t96_01_,
        't01':   _lib.t01_01_,
        't01s':  _lib.t01_s_,
        'ts04':  _lib.t04_s_,
        'ts07':  _lib.ts07d_2015_,
    }

    # Build geopack-interface adapters for EXNAME.
    # RHAND_08 (inside TRACE_08) calls EXNAME with 9 by-reference arguments:
    #   EXNAME(IOPT, PARMOD, PS, X, Y, Z, BX, BY, BZ)
    # IRBEM's models have different, shorter signatures, so we wrap them.
    global _GP_EXT_FUNCS
    _GP_EXT_FUNCS = _build_geopack_ext_adapters(_lib)


def _build_geopack_ext_adapters(lib):
    """Return a dict of CFUNCTYPE adapters that satisfy the 9-arg EXNAME
    interface expected by RHAND_08, delegating to each IRBEM field model.

    RHAND_08 calls:  EXNAME(IOPT, PARMOD, PS, X, Y, Z, BX_out, BY_out, BZ_out)
    All arguments are Fortran by-reference (C pointers).
    """
    t89c   = lib.t89c_       # T89C(IOPT, X, Y, Z, BX, BY, BZ)          – 7 args
    t96    = lib.t96_01_     # T96_01(PARMOD, X, Y, Z, BX, BY, BZ)       – 7 args
    t01    = lib.t01_01_     # T01_01(PARMOD, X, Y, Z, BX, BY, BZ)       – 7 args
    t01s   = lib.t01_s_      # T01_S(PARMOD, X, Y, Z, BX, BY, BZ)        – 7 args
    t04s   = lib.t04_s_      # T04_S(PARMOD, X, Y, Z, BX, BY, BZ)        – 7 args
    ts07   = lib.ts07d_2015_ # TS07D_2015(X, Y, Z, BX, BY, BZ)           – 6 args
    dip    = lib.dip_08_     # DIP_08(X, Y, Z, BX, BY, BZ)               – 6 args

    def _call_t89c(iopt_p, parmod_p, ps_p, x_p, y_p, z_p, bx_p, by_p, bz_p):
        iopt_v = iopt_p[0]
        x_v, y_v, z_v = x_p[0], y_p[0], z_p[0]
        bx_c, by_c, bz_c = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        t89c(ctypes.byref(ctypes.c_int(iopt_v)),
             ctypes.byref(ctypes.c_double(x_v)),
             ctypes.byref(ctypes.c_double(y_v)),
             ctypes.byref(ctypes.c_double(z_v)),
             ctypes.byref(bx_c), ctypes.byref(by_c), ctypes.byref(bz_c))
        bx_p[0], by_p[0], bz_p[0] = bx_c.value, by_c.value, bz_c.value

    def _call_parmod(func):
        """Return a 9-arg adapter for a PARMOD-based model (T96, T01, T04)."""
        def _impl(iopt_p, parmod_p, ps_p, x_p, y_p, z_p, bx_p, by_p, bz_p):
            parmod = (ctypes.c_double * 10)(*[parmod_p[i] for i in range(10)])
            x_v, y_v, z_v = x_p[0], y_p[0], z_p[0]
            bx_c, by_c, bz_c = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            func(parmod,
                 ctypes.byref(ctypes.c_double(x_v)),
                 ctypes.byref(ctypes.c_double(y_v)),
                 ctypes.byref(ctypes.c_double(z_v)),
                 ctypes.byref(bx_c), ctypes.byref(by_c), ctypes.byref(bz_c))
            bx_p[0], by_p[0], bz_p[0] = bx_c.value, by_c.value, bz_c.value
        return _impl

    def _call_xyz6(func):
        """Return a 9-arg adapter for a 6-arg model (X, Y, Z, BX, BY, BZ)."""
        def _impl(iopt_p, parmod_p, ps_p, x_p, y_p, z_p, bx_p, by_p, bz_p):
            x_v, y_v, z_v = x_p[0], y_p[0], z_p[0]
            bx_c, by_c, bz_c = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            func(ctypes.byref(ctypes.c_double(x_v)),
                 ctypes.byref(ctypes.c_double(y_v)),
                 ctypes.byref(ctypes.c_double(z_v)),
                 ctypes.byref(bx_c), ctypes.byref(by_c), ctypes.byref(bz_c))
            bx_p[0], by_p[0], bz_p[0] = bx_c.value, by_c.value, bz_c.value
        return _impl

    def _zero_ext(iopt_p, parmod_p, ps_p, x_p, y_p, z_p, bx_p, by_p, bz_p):
        """Zero external field – used when 'igrf' is passed as exname."""
        bx_p[0], by_p[0], bz_p[0] = 0.0, 0.0, 0.0

    # Wrap each callable in _EXNAME_CFUNCTYPE so Fortran can call it via
    # a procedure pointer.  The CFUNCTYPE object must stay alive for the
    # duration of any TRACE_08 call, so we keep them in this dict.
    return {
        't89':  _EXNAME_CFUNCTYPE(_call_t89c),
        't96':  _EXNAME_CFUNCTYPE(_call_parmod(t96)),
        't01':  _EXNAME_CFUNCTYPE(_call_parmod(t01)),
        't01s': _EXNAME_CFUNCTYPE(_call_parmod(t01s)),
        'ts04': _EXNAME_CFUNCTYPE(_call_parmod(t04s)),
        'ts07': _EXNAME_CFUNCTYPE(_call_xyz6(ts07)),
        'dip':  _EXNAME_CFUNCTYPE(_call_xyz6(dip)),
        'igrf': _EXNAME_CFUNCTYPE(_zero_ext),
    }


def _check_lib():
    if _lib is None:
        raise RuntimeError(
            "IRBEM library not loaded. Call geopack.init() before using "
            "any other function."
        )


def _check_recalc():
    if not _recalc_called:
        raise RuntimeError(
            "recalc() has not been called. Invoke recalc() with a valid "
            "date/time before calling field or coordinate-transform functions."
        )


# --------------------------------------------------------------------------- #
# Time helper
# --------------------------------------------------------------------------- #
def _unpack_time(t):
    """Return (year, doy, hour, minute, second) from a datetime-like object."""
    if isinstance(t, np.datetime64):
        ts = int(t.astype('datetime64[s]').astype('int64'))
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=ts)
    elif isinstance(t, datetime.datetime):
        dt = t
    else:
        # Try str or other formats via numpy
        ts = int(np.datetime64(str(t), 's').astype('int64'))
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=ts)
    return dt.year, dt.timetuple().tm_yday, dt.hour, dt.minute, dt.second


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def recalc(t, vgsex=-400.0, vgsey=0.0, vgsez=0.0):
    """
    Compute internal transformation matrices for date/time *t*.

    **Must be called before any field or coordinate-transform function.**

    Parameters
    ----------
    t : numpy.datetime64 or datetime.datetime
        Universal time.
    vgsex, vgsey, vgsez : float
        Solar wind velocity components in GSE coordinates (km/s).
        Default (-400, 0, 0) – purely antisunward flow.
        These components define the GSW x-axis direction (antiparallel to
        the solar wind flow vector). The dipole tilt angle PSI is computed
        as arcsin(DIP · EXGSW), so non-zero VGSEY or VGSEZ will shift the
        GSW frame away from GSM and change the returned PSI value.  When
        VGSEY = VGSEZ = 0 the GSW and GSM frames are identical.

    Returns
    -------
    float
        Dipole tilt angle PSI (degrees) in the GSW coordinate system.
        Positive when the north magnetic pole tilts toward the Sun.
        Equal to the standard GSM tilt when VGSEY = VGSEZ = 0.
    """
    _check_lib()
    year, doy, hour, minute, sec = _unpack_time(t)
    _lib.recalc_08_(
        ctypes.byref(ctypes.c_int(year)),
        ctypes.byref(ctypes.c_int(doy)),
        ctypes.byref(ctypes.c_int(hour)),
        ctypes.byref(ctypes.c_int(minute)),
        ctypes.byref(ctypes.c_int(sec)),
        ctypes.byref(ctypes.c_double(vgsex)),
        ctypes.byref(ctypes.c_double(vgsey)),
        ctypes.byref(ctypes.c_double(vgsez)),
    )
    global _recalc_called
    _recalc_called = True
    # PSI is element 15 (0-indexed) of the GEOPACK1 common block
    block = (ctypes.c_double * 34).in_dll(_lib, 'geopack1_')
    tilt = float(np.degrees(block[15]))
    # Populate the IRBEM /dip_ang/ common block so that IRBEM Tsyganenko
    # models (T89C, T96_01, T01_01, T01_S, T04_s) read the correct tilt
    # angle (in degrees) when called directly from the Python interface.
    dip_ang = (ctypes.c_double * 1).in_dll(_lib, 'dip_ang_')
    dip_ang[0] = tilt
    return tilt


# --------------------------------------------------------------------------- #
# Internal field functions
# --------------------------------------------------------------------------- #
def igrf_geo(r, theta, phi):
    """
    IGRF field in geocentric spherical coordinates.

    Parameters
    ----------
    r : float
        Geocentric distance (Earth radii).
    theta : float
        Geocentric colatitude (radians, 0 = north pole).
    phi : float
        Geographic longitude (radians, eastward from Greenwich).

    Returns
    -------
    tuple of float
        (Br, Btheta, Bphi) in nT.
    """
    _check_lib()
    _check_recalc()
    br = ctypes.c_double()
    btheta = ctypes.c_double()
    bphi = ctypes.c_double()
    _lib.igrf_geo_08_(
        ctypes.byref(ctypes.c_double(r)),
        ctypes.byref(ctypes.c_double(theta)),
        ctypes.byref(ctypes.c_double(phi)),
        ctypes.byref(br),
        ctypes.byref(btheta),
        ctypes.byref(bphi),
    )
    return br.value, btheta.value, bphi.value


def igrf_gsw(x, y, z):
    """
    IGRF field in GSW Cartesian coordinates.

    Parameters
    ----------
    x, y, z : float
        Position in GSW coordinates (Earth radii).

    Returns
    -------
    tuple of float
        (Bx, By, Bz) in nT.
    """
    _check_lib()
    _check_recalc()
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    _lib.igrf_gsw_08_(
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(bx),
        ctypes.byref(by),
        ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


def dip(x, y, z):
    """
    Pure dipole field in GSW Cartesian coordinates.

    Parameters
    ----------
    x, y, z : float
        Position in GSW coordinates (Earth radii).

    Returns
    -------
    tuple of float
        (Bx, By, Bz) in nT.
    """
    _check_lib()
    _check_recalc()
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    _lib.dip_08_(
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(bx),
        ctypes.byref(by),
        ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


# --------------------------------------------------------------------------- #
# Coordinate conversion helpers
# --------------------------------------------------------------------------- #
def sphcar(v1, v2, v3, to_rect=True):
    """
    Convert between spherical and Cartesian coordinates.

    Parameters
    ----------
    v1, v2, v3 : float
        If *to_rect*: (r [RE], theta [rad], phi [rad]).
        If not *to_rect*: (x, y, z) in Earth radii.
    to_rect : bool
        True → sphere→rect;  False → rect→sphere.

    Returns
    -------
    tuple of float
        (x, y, z) [RE] when to_rect=True; (r, theta, phi) when False.
    """
    _check_lib()
    # Fortran signature: SPHCAR_08(R, THETA, PHI, X, Y, Z, J)
    # J>0: input r,theta,phi (args 1-3) → output x,y,z (args 4-6)
    # J<0: input x,y,z (args 4-6)      → output r,theta,phi (args 1-3)
    j = ctypes.c_int(1 if to_rect else -1)
    if to_rect:
        r     = ctypes.c_double(v1)
        theta = ctypes.c_double(v2)
        phi   = ctypes.c_double(v3)
        x = ctypes.c_double()
        y = ctypes.c_double()
        z = ctypes.c_double()
    else:
        r     = ctypes.c_double()
        theta = ctypes.c_double()
        phi   = ctypes.c_double()
        x = ctypes.c_double(v1)
        y = ctypes.c_double(v2)
        z = ctypes.c_double(v3)
    _lib.sphcar_08_(
        ctypes.byref(r), ctypes.byref(theta), ctypes.byref(phi),
        ctypes.byref(x), ctypes.byref(y), ctypes.byref(z),
        ctypes.byref(j),
    )
    if to_rect:
        return x.value, y.value, z.value
    else:
        return r.value, theta.value, phi.value


def bspcar(theta, phi, br, btheta, bphi):
    """
    Convert field vector from spherical to Cartesian components.

    Parameters
    ----------
    theta, phi : float
        Colatitude and longitude at the field point (radians).
    br, btheta, bphi : float
        Spherical field components (nT).

    Returns
    -------
    tuple of float
        (Bx, By, Bz) in nT.
    """
    _check_lib()
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    _lib.bspcar_08_(
        ctypes.byref(ctypes.c_double(theta)),
        ctypes.byref(ctypes.c_double(phi)),
        ctypes.byref(ctypes.c_double(br)),
        ctypes.byref(ctypes.c_double(btheta)),
        ctypes.byref(ctypes.c_double(bphi)),
        ctypes.byref(bx),
        ctypes.byref(by),
        ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


def bcarsp(x, y, z, bx, by, bz):
    """
    Convert field vector from Cartesian to spherical components.

    Parameters
    ----------
    x, y, z : float
        Position in Earth radii.
    bx, by, bz : float
        Cartesian field components (nT).

    Returns
    -------
    tuple of float
        (Br, Btheta, Bphi) in nT.
    """
    _check_lib()
    br = ctypes.c_double()
    btheta = ctypes.c_double()
    bphi = ctypes.c_double()
    _lib.bcarsp_08_(
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(ctypes.c_double(bx)),
        ctypes.byref(ctypes.c_double(by)),
        ctypes.byref(ctypes.c_double(bz)),
        ctypes.byref(br),
        ctypes.byref(btheta),
        ctypes.byref(bphi),
    )
    return br.value, btheta.value, bphi.value


def _apply_transform(func, j, x, y, z):
    """Apply one elementary Geopack coordinate transform.

    All Geopack transforms share the signature:
        SUBROUTINE XXXYYY_08(A1, A2, A3, B1, B2, B3, J)
    where J>0 reads input from A1,A2,A3 and writes output to B1,B2,B3,
    and J<0 reads input from B1,B2,B3 and writes output to A1,A2,A3.
    """
    jc = ctypes.c_int(j)
    if j > 0:
        a1 = ctypes.c_double(x)
        a2 = ctypes.c_double(y)
        a3 = ctypes.c_double(z)
        b1 = ctypes.c_double()
        b2 = ctypes.c_double()
        b3 = ctypes.c_double()
    else:
        a1 = ctypes.c_double()
        a2 = ctypes.c_double()
        a3 = ctypes.c_double()
        b1 = ctypes.c_double(x)
        b2 = ctypes.c_double(y)
        b3 = ctypes.c_double(z)
    func(
        ctypes.byref(a1), ctypes.byref(a2), ctypes.byref(a3),
        ctypes.byref(b1), ctypes.byref(b2), ctypes.byref(b3),
        ctypes.byref(jc),
    )
    if j > 0:
        return b1.value, b2.value, b3.value
    else:
        return a1.value, a2.value, a3.value


def _find_path(src, dst):
    """BFS over the transform graph to find a chain from *src* to *dst*."""
    if src == dst:
        return [src]
    visited = {src}
    queue = deque([[src]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        for (a, b) in _TRANSFORMS:
            if a == node and b not in visited:
                new_path = path + [b]
                if b == dst:
                    return new_path
                visited.add(b)
                queue.append(new_path)
    raise ValueError(f"No transform path found: '{src}' → '{dst}'")


def conv_coord(x1, x2, x3, from_sys, to_sys):
    """
    Convert Cartesian coordinates between Geopack coordinate systems.

    Supported systems: ``'GEO'``, ``'MAG'``, ``'GEI'``, ``'SM'``,
    ``'GSW'``, ``'GSM'`` (alias for GSW), ``'GSE'``.

    ``recalc()`` must have been called before this function.

    Parameters
    ----------
    x1, x2, x3 : float
        Input position in Earth radii.
    from_sys : str
        Source coordinate system.
    to_sys : str
        Target coordinate system.

    Returns
    -------
    tuple of float
        (d1, d2, d3) output position in Earth radii.
    """
    _check_lib()
    _check_recalc()
    src = from_sys.upper().replace('GSM', 'GSW')
    dst = to_sys.upper().replace('GSM', 'GSW')
    if src == dst:
        return float(x1), float(x2), float(x3)
    path = _find_path(src, dst)
    x, y, z = float(x1), float(x2), float(x3)
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        func, j = _TRANSFORMS[(a, b)]
        x, y, z = _apply_transform(func, j, x, y, z)
    return x, y, z


# --------------------------------------------------------------------------- #
# External field models
# --------------------------------------------------------------------------- #
def t89(iopt, x, y, z):
    """
    Tsyganenko 1989 external field (T89c).

    Parameters
    ----------
    iopt : int
        Kp-related activity index (1 = Kp < 1 … 7 = Kp ≥ 6).
    x, y, z : float
        Position in GSW coordinates (Earth radii).

    Returns
    -------
    tuple of float
        (Bx, By, Bz) external field contribution in nT.
    """
    _check_lib()
    _check_recalc()
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    _lib.t89c_(
        ctypes.byref(ctypes.c_int(int(iopt))),
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(bx),
        ctypes.byref(by),
        ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


def t96(parmod, x, y, z):
    """
    Tsyganenko 1996 external field.

    Parameters
    ----------
    parmod : array-like, length 10
        ``[Pdyn (nPa), Dst (nT), ByIMF (nT), BzIMF (nT), 0, 0, 0, 0, 0, 0]``
    x, y, z : float
        Position in GSW coordinates (Earth radii).

    Returns
    -------
    tuple of float
        (Bx, By, Bz) in nT.
    """
    _check_lib()
    _check_recalc()
    pm = (ctypes.c_double * 10)(*[float(v) for v in parmod])
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    _lib.t96_01_(pm,
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(bx), ctypes.byref(by), ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


def t01(parmod, x, y, z):
    """
    Tsyganenko 2001 external field.

    Parameters
    ----------
    parmod : array-like, length 10
        ``[Pdyn, Dst, ByIMF, BzIMF, G1, G2, 0, 0, 0, 0]``
    x, y, z : float
        Position in GSW coordinates (Earth radii).

    Returns
    -------
    tuple of float
        (Bx, By, Bz) in nT.
    """
    _check_lib()
    _check_recalc()
    pm = (ctypes.c_double * 10)(*[float(v) for v in parmod])
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    _lib.t01_01_(pm,
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(bx), ctypes.byref(by), ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


def t01s(parmod, x, y, z):
    """
    Tsyganenko 2001 storm-time external field.

    Parameters
    ----------
    parmod : array-like, length 10
        ``[Pdyn, Dst, ByIMF, BzIMF, G2, G3, 0, 0, 0, 0]``
    x, y, z : float
        Position in GSW coordinates (Earth radii).

    Returns
    -------
    tuple of float
        (Bx, By, Bz) in nT.
    """
    _check_lib()
    _check_recalc()
    pm = (ctypes.c_double * 10)(*[float(v) for v in parmod])
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    _lib.t01_s_(pm,
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(bx), ctypes.byref(by), ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


def ts04(parmod, x, y, z):
    """
    Tsyganenko-Sitnov 2004 external field.

    Parameters
    ----------
    parmod : array-like, length 10
        ``[Pdyn, Dst, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6]``
    x, y, z : float
        Position in GSW coordinates (Earth radii).

    Returns
    -------
    tuple of float
        (Bx, By, Bz) in nT.
    """
    _check_lib()
    _check_recalc()
    pm = (ctypes.c_double * 10)(*[float(v) for v in parmod])
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    _lib.t04_s_(pm,
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(bx), ctypes.byref(by), ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


def ts07(x, y, z, version='2015'):
    """
    Tsyganenko-Sitnov 2007 external field (TS07D).

    Requires TS07D coefficient files to be loaded (see setup_ts07d_files.sh).

    Parameters
    ----------
    x, y, z : float
        Position in GSW coordinates (Earth radii).
    version : {'2015', '2017'}
        Model version. Default ``'2015'``.

    Returns
    -------
    tuple of float
        (Bx, By, Bz) in nT.
    """
    _check_lib()
    _check_recalc()
    bx = ctypes.c_double()
    by = ctypes.c_double()
    bz = ctypes.c_double()
    func = _lib.ts07d_july_2017_ if version == '2017' else _lib.ts07d_2015_
    func(
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(bx),
        ctypes.byref(by),
        ctypes.byref(bz),
    )
    return bx.value, by.value, bz.value


# --------------------------------------------------------------------------- #
# Magnetopause
# --------------------------------------------------------------------------- #
def t96_mgnp(xn_pd, vel, x, y, z):
    """
    Find the T96 magnetopause crossing point nearest to (x, y, z).

    Parameters
    ----------
    xn_pd : float
        Solar wind proton number density (cm⁻³).
    vel : float
        Solar wind speed (km/s; use negative for antisunward flow, e.g. -400).
    x, y, z : float
        Query point in GSW coordinates (Earth radii).

    Returns
    -------
    tuple
        ``(x_mgnp, y_mgnp, z_mgnp, dist, id)``
        where *id* = +1 if inside the magnetopause, -1 if outside.
    """
    _check_lib()
    xm = ctypes.c_double()
    ym = ctypes.c_double()
    zm = ctypes.c_double()
    dist = ctypes.c_double()
    id_ = ctypes.c_int()
    _lib.t96_mgnp_08_(
        ctypes.byref(ctypes.c_double(xn_pd)),
        ctypes.byref(ctypes.c_double(vel)),
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(xm), ctypes.byref(ym), ctypes.byref(zm),
        ctypes.byref(dist), ctypes.byref(id_),
    )
    return xm.value, ym.value, zm.value, dist.value, id_.value


def shuetal_mgnp(xn_pd, vel, bzimf, x, y, z):
    """
    Find the Shue et al. (1997) magnetopause crossing point nearest to (x, y, z).

    Parameters
    ----------
    xn_pd : float
        Solar wind proton number density (cm⁻³).
    vel : float
        Solar wind speed (km/s; use negative for antisunward).
    bzimf : float
        IMF Bz component (nT).
    x, y, z : float
        Query point in GSW coordinates (Earth radii).

    Returns
    -------
    tuple
        ``(x_mgnp, y_mgnp, z_mgnp, dist, id)``
        where *id* = +1 if inside the magnetopause, -1 if outside.
    """
    _check_lib()
    xm = ctypes.c_double()
    ym = ctypes.c_double()
    zm = ctypes.c_double()
    dist = ctypes.c_double()
    id_ = ctypes.c_int()
    _lib.shuetal_mgnp_08_(
        ctypes.byref(ctypes.c_double(xn_pd)),
        ctypes.byref(ctypes.c_double(vel)),
        ctypes.byref(ctypes.c_double(bzimf)),
        ctypes.byref(ctypes.c_double(x)),
        ctypes.byref(ctypes.c_double(y)),
        ctypes.byref(ctypes.c_double(z)),
        ctypes.byref(xm), ctypes.byref(ym), ctypes.byref(zm),
        ctypes.byref(dist), ctypes.byref(id_),
    )
    return xm.value, ym.value, zm.value, dist.value, id_.value


# --------------------------------------------------------------------------- #
# Field-line tracing
# --------------------------------------------------------------------------- #
def trace(xi, yi, zi, dir, rlim=60.0, r0=1.0, dsmax=0.5, err=0.0001,
          iopt=1, parmod=None, exname='igrf', inname='igrf'):
    """
    Trace a magnetic field line using Geopack's ``TRACE_08``.

    ``recalc()`` must be called before this function.

    Parameters
    ----------
    xi, yi, zi : float
        Starting position in GSW coordinates (Earth radii).
    dir : float
        Tracing direction: +1 (along B) or -1 (opposite to B).
    rlim : float
        Outer boundary (Earth radii). Default 60.
    r0 : float
        Inner boundary (Earth radii). Default 1 (Earth surface).
    dsmax : float
        Maximum step size (Earth radii). Default 0.5.
    err : float
        Runge-Kutta tolerance. Default 0.0001.
    iopt : int
        Integer option for external model (e.g., Kp for T89). Default 1.
    parmod : array-like of length 10, optional
        Parameter array for external model. Default all zeros.
    exname : str
        External field model. One of  't89','t96', 't01', 't01s', 'ts04', 'ts07'.
    inname : str
        Internal field model: 'igrf' (default) or 'dip'.

    Returns
    -------
    dict
        ``'xf'``, ``'yf'``, ``'zf'`` – endpoint coordinates (Earth radii).
        ``'xx'``, ``'yy'``, ``'zz'`` – arrays of field-line positions.
        ``'npts'`` – number of points along the field line.
    """
    _check_lib()
    _check_recalc()
    if parmod is None:
        parmod = [0.0] * 10
    pm = (ctypes.c_double * 10)(*[float(v) for v in parmod])

    LMAX = 3000
    xx_arr = (ctypes.c_double * LMAX)()
    yy_arr = (ctypes.c_double * LMAX)()
    zz_arr = (ctypes.c_double * LMAX)()
    xf = ctypes.c_double()
    yf = ctypes.c_double()
    zf = ctypes.c_double()
    l_out = ctypes.c_int()
    lmax_c = ctypes.c_int(LMAX)

    # EXNAME needs the 9-arg geopack interface; use the CFUNCTYPE adapter.
    # INNAME is IGRF_GSW_08 or DIP_08, which already have the correct 6-arg
    # interface that RHAND_08 calls: INNAME(X, Y, Z, HX, HY, HZ).
    ex_adapter = _GP_EXT_FUNCS[exname.lower()]  # keep alive during call
    ex_ptr = ctypes.cast(ex_adapter, ctypes.c_void_p)
    in_ptr = ctypes.cast(_MODEL_FUNCS[inname.lower()], ctypes.c_void_p)

    _lib.trace_08_(
        ctypes.byref(ctypes.c_double(xi)),
        ctypes.byref(ctypes.c_double(yi)),
        ctypes.byref(ctypes.c_double(zi)),
        ctypes.byref(ctypes.c_double(dir)),
        ctypes.byref(ctypes.c_double(dsmax)),
        ctypes.byref(ctypes.c_double(err)),
        ctypes.byref(ctypes.c_double(rlim)),
        ctypes.byref(ctypes.c_double(r0)),
        ctypes.byref(ctypes.c_int(int(iopt))),
        pm,
        ex_ptr,
        in_ptr,
        ctypes.byref(xf),
        ctypes.byref(yf),
        ctypes.byref(zf),
        xx_arr,
        yy_arr,
        zz_arr,
        ctypes.byref(l_out),
        ctypes.byref(lmax_c),
    )
    n = l_out.value
    return {
        'xf': xf.value, 'yf': yf.value, 'zf': zf.value,
        'xx': np.array(xx_arr[:n]),
        'yy': np.array(yy_arr[:n]),
        'zz': np.array(zz_arr[:n]),
        'npts': n,
    }


def _prep_maginput(maginput):
    """Build the 25-element maginput ctypes array from a dict."""
    arr = (ctypes.c_double * 25)()
    for i in range(25):
        arr[i] = -9999.0
    if maginput:
        for i, key in enumerate(_ORDERED_MAGINPUT_KEYS):
            if key in maginput:
                arr[i] = float(maginput[key])
    return arr


def trace_field_line(t, x, y, z, kext=4, maginput=None, sysaxes=3, options=None):
    """
    Trace a full magnetic field line using IRBEM's ``trace_field_line1``.

    Unlike the low-level :func:`trace`, this function handles coordinate
    conversion and magnetic-field initialisation internally via the IRBEM
    high-level API.  It does **not** require a prior :func:`recalc` call.

    Parameters
    ----------
    t : numpy.datetime64 or datetime.datetime
        Universal time.
    x, y, z : float
        Position in the coordinate system given by *sysaxes*.
    kext : int or str
        External field model.  Integer (IRBEM convention) or one of
        ``'none'``/``'dipole'`` (0), ``'t89'`` (4), ``'t96'`` (7),
        ``'t01'`` (9), ``'t01s'`` (10), ``'ts04'`` (11), ``'ts07'`` (13).
        Default 4 (T89).
    maginput : dict, optional
        Model parameters keyed by ``'Kp'``, ``'Dst'``, ``'Pdyn'``,
        ``'ByIMF'``, ``'BzIMF'``, ``'G1'``, ``'G2'``, ``'W1'``…``'W6'``, etc.
    sysaxes : int
        IRBEM input coordinate system:
        0=GDZ, 1=GEO, 2=GSM, 3=GSE, 4=SM, 5=GEI, 6=MAG.  Default 3 (GSE).
    options : list of 5 ints, optional
        IRBEM options array.  Default ``[0, 0, 0, 0, 0]``.

    Returns
    -------
    dict
        ``'Lm'`` – McIlwain L shell.
        ``'Blocal'`` – local field magnitude along the line (nT), shape (Nposit,).
        ``'Bmin'`` – minimum field magnitude on the line (nT).
        ``'XJ'`` – second adiabatic invariant (RE).
        ``'POSIT'`` – field-line positions in GEO (RE), shape (Nposit, 3).
        ``'Nposit'`` – number of points along the field line.
    """
    _check_lib()
    if options is None:
        options = [0, 0, 0, 0, 0]
    if isinstance(kext, str):
        kext = _KEXT_MAP.get(kext.lower(), 4)

    year, doy, hour, minute, sec = _unpack_time(t)
    ut = float(3600 * hour + 60 * minute + sec)

    kext_c    = ctypes.c_int(int(kext))
    opts_c    = (ctypes.c_int * 5)(*[int(o) for o in options])
    sysaxes_c = ctypes.c_int(int(sysaxes))
    iyear_c   = ctypes.c_int(year)
    idoy_c    = ctypes.c_int(doy)
    ut_c      = ctypes.c_double(ut)
    x1_c      = ctypes.c_double(float(x))
    x2_c      = ctypes.c_double(float(y))
    x3_c      = ctypes.c_double(float(z))
    maginput_c = _prep_maginput(maginput)

    lm_c     = ctypes.c_double()
    bmin_c   = ctypes.c_double()
    xj_c     = ctypes.c_double()
    # Fortran: BLOCAL(3000)
    blocal_c = (ctypes.c_double * 3000)()
    # Fortran: posit(3, 3000) → ctypes layout (3000, 3)
    posit_c  = ((ctypes.c_double * 3) * 3000)()
    ind_c    = ctypes.c_int()

    _lib.trace_field_line1_(
        ctypes.byref(kext_c),
        ctypes.byref(opts_c),
        ctypes.byref(sysaxes_c),
        ctypes.byref(iyear_c),
        ctypes.byref(idoy_c),
        ctypes.byref(ut_c),
        ctypes.byref(x1_c),
        ctypes.byref(x2_c),
        ctypes.byref(x3_c),
        ctypes.byref(maginput_c),
        ctypes.byref(lm_c),
        ctypes.byref(blocal_c),
        ctypes.byref(bmin_c),
        ctypes.byref(xj_c),
        ctypes.byref(posit_c),
        ctypes.byref(ind_c),
    )

    n = ind_c.value
    npts = max(n, 0)
    return {
        'Lm':     lm_c.value,
        'Blocal': np.array(blocal_c[:npts]),
        'Bmin':   bmin_c.value,
        'XJ':     xj_c.value,
        'POSIT':  np.array(posit_c[:npts]),   # shape (npts, 3)
        'Nposit': n,
    }


# --------------------------------------------------------------------------- #
# Auto-init at import time (graceful failure if library not built yet)
# --------------------------------------------------------------------------- #
try:
    init()
except FileNotFoundError:
    pass  # Library not built; user must call init(path=...) explicitly
