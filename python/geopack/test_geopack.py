"""
Basic correctness tests for the Python geopack wrapper.

Run with:  python -m pytest test_geopack.py -v
       or: python test_geopack.py
"""

import functools
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

# Allow running from the geopack/ directory without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geopack

# Reference epoch used throughout the tests
T0 = np.datetime64('2000-01-11T17:00:00')


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _recalc(t=T0):
    geopack.recalc(t)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_recalc():
    """recalc() should complete without error for a valid epoch."""
    tilt1 = geopack.recalc(T0)
    
    # Call again with explicit solar wind vector
    tilt2 = geopack.recalc(T0, vgsex=-450.0, vgsey=5.0, vgsez=-2.0)

    # Call again with explicit solar wind vector
    tilt3 = geopack.recalc(T0, vgsex=-450.0, vgsey=0.0, vgsez=0.0)
    tilt4 = geopack.recalc(T0, vgsex=-550.0, vgsey=0.0, vgsez=0.0)


    print(tilt1, tilt2, tilt3, tilt4)
    assert tilt1 == pytest.approx(-11.380684)
    assert tilt2 == pytest.approx(-11.228203)
    assert tilt3 == pytest.approx(-11.380684)
    assert tilt4 == pytest.approx(-11.380684)


def test_igrf_geo():
    """IGRF field in geocentric spherical coords should be physically sane."""
    _recalc()
    # Evaluate at r=1 RE, theta=pi/2 (equator), phi=0
    br, bt, bp = geopack.igrf_geo(1.0, math.pi / 2, 0.0)
    # The main field at Earth's surface is ~25 000 – 65 000 nT
    bmag = math.sqrt(br**2 + bt**2 + bp**2)
    assert 20_000 < bmag < 70_000, f"IGRF magnitude {bmag:.0f} nT out of expected range"


def test_igrf_gsw():
    """IGRF in GSW Cartesian coords should be non-zero at Earth's surface."""
    _recalc()
    bx, by, bz = geopack.igrf_gsw(1.0, 0.0, 0.0)
    bmag = math.sqrt(bx**2 + by**2 + bz**2)
    assert bmag > 1000, f"IGRF |B| {bmag:.0f} nT suspiciously small"


def test_dip():
    """Dipole field should have physically reasonable magnitude at 4 RE."""
    _recalc()
    bx, by, bz = geopack.dip(4.0, 0.0, 0.0)
    bmag = math.sqrt(bx**2 + by**2 + bz**2)
    # Earth surface field ~30 000 nT; scales as 1/r³ → at 4 RE: ~30000/64 ≈ 470 nT
    assert 100 < bmag < 2000, f"Dipole |B| at 4 RE = {bmag:.0f} nT out of expected range"
    # On the GSW x-axis the dipole is symmetric: B(-x,0,0) == B(x,0,0) for Bz
    bx2, by2, bz2 = geopack.dip(-4.0, 0.0, 0.0)
    assert abs(bz - bz2) < 1.0, f"Dipole Bz not same at (±4,0,0): {bz:.2f} vs {bz2:.2f}"


def test_sphcar_roundtrip():
    """Sphere→rect→sphere should recover original coordinates."""
    _recalc()
    r0, theta0, phi0 = 3.5, 1.2, 0.7
    x, y, z = geopack.sphcar(r0, theta0, phi0, to_rect=True)
    r1, theta1, phi1 = geopack.sphcar(x, y, z, to_rect=False)
    assert abs(r1 - r0) < 1e-10, f"r round-trip error: {r1} vs {r0}"
    assert abs(theta1 - theta0) < 1e-10, f"theta round-trip error: {theta1} vs {theta0}"
    assert abs(phi1 - phi0) < 1e-10, f"phi round-trip error: {phi1} vs {phi0}"


def test_bspcar_bcarsp_roundtrip():
    """bspcar / bcarsp should be inverse of each other."""
    _recalc()
    theta, phi = 1.1, 0.5
    br0, bt0, bp0 = 100.0, -200.0, 50.0
    bx, by, bz = geopack.bspcar(theta, phi, br0, bt0, bp0)
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    br1, bt1, bp1 = geopack.bcarsp(x, y, z, bx, by, bz)
    assert abs(br1 - br0) < 1e-8, f"br round-trip: {br1} vs {br0}"
    assert abs(bt1 - bt0) < 1e-8, f"btheta round-trip: {bt1} vs {bt0}"
    assert abs(bp1 - bp0) < 1e-8, f"bphi round-trip: {bp1} vs {bp0}"


def test_conv_coord_identity():
    """Converting to the same system should return the original coordinates."""
    _recalc()
    x, y, z = 3.0, 1.5, -0.5
    x2, y2, z2 = geopack.conv_coord(x, y, z, 'GEO', 'GEO')
    assert abs(x2 - x) < 1e-12
    assert abs(y2 - y) < 1e-12
    assert abs(z2 - z) < 1e-12


def test_conv_coord_geo_gsm_roundtrip():
    """GEO → GSM → GEO round-trip should recover original coordinates."""
    _recalc()
    x0, y0, z0 = 3.0, 1.5, -0.5
    xg, yg, zg = geopack.conv_coord(x0, y0, z0, 'GEO', 'GSM')
    x1, y1, z1 = geopack.conv_coord(xg, yg, zg, 'GSM', 'GEO')
    assert abs(x1 - x0) < 1e-8, f"GEO→GSM→GEO x: {x1} vs {x0}"
    assert abs(y1 - y0) < 1e-8, f"GEO→GSM→GEO y: {y1} vs {y0}"
    assert abs(z1 - z0) < 1e-8, f"GEO→GSM→GEO z: {z1} vs {z0}"


def test_conv_coord_geo_gse_roundtrip():
    """GEO → GSE → GEO round-trip."""
    _recalc()
    x0, y0, z0 = 2.0, -1.0, 0.5
    xg, yg, zg = geopack.conv_coord(x0, y0, z0, 'GEO', 'GSE')
    x1, y1, z1 = geopack.conv_coord(xg, yg, zg, 'GSE', 'GEO')
    assert abs(x1 - x0) < 1e-8
    assert abs(y1 - y0) < 1e-8
    assert abs(z1 - z0) < 1e-8


def test_conv_coord_preserves_length():
    """Coordinate rotations must preserve vector magnitude."""
    _recalc()
    x0, y0, z0 = 4.0, 2.0, -1.0
    r0 = math.sqrt(x0**2 + y0**2 + z0**2)
    for dst in ('MAG', 'SM', 'GSM', 'GSE', 'GEI'):
        xd, yd, zd = geopack.conv_coord(x0, y0, z0, 'GEO', dst)
        rd = math.sqrt(xd**2 + yd**2 + zd**2)
        assert abs(rd - r0) < 1e-6 * r0, f"GEO→{dst} changed magnitude: {rd} vs {r0}"


def test_t89():
    """T89 should return a physically plausible external field."""
    _recalc()
    bx, by, bz = geopack.t89(3, -4.0, 2.0, 1.0)
    bmag = math.sqrt(bx**2 + by**2 + bz**2)
    # External field at ~4 RE should be order 10–200 nT
    assert 1 < bmag < 1000, f"T89 |B| = {bmag:.1f} nT seems wrong"


def test_t96():
    """T96 should run without error."""
    _recalc()
    parmod = [2.0, -10.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0]
    bx, by, bz = geopack.t96(parmod, -4.0, 2.0, 1.0)
    bmag = math.sqrt(bx**2 + by**2 + bz**2)
    assert bmag > 0


def test_t96_mgnp_inside():
    """A point well inside the magnetopause (near Earth) should be flagged inside (id=+1)."""
    _recalc()
    # GSW: positive x = sunward. 3 RE sunward is deep inside (nose ~10 RE).
    xm, ym, zm, dist, id_ = geopack.t96_mgnp(5.0, -400.0, 3.0, 0.0, 0.0)
    assert id_ == 1, f"Expected inside (+1), got {id_}"


def test_t96_mgnp_outside():
    """A point well beyond the subsolar magnetopause should be outside (id=-1)."""
    _recalc()
    # 25 RE sunward is well beyond the ~10 RE subsolar nose.
    xm, ym, zm, dist, id_ = geopack.t96_mgnp(5.0, -400.0, 25.0, 0.0, 0.0)
    assert id_ == -1, f"Expected outside (-1), got {id_}"


def test_shuetal_mgnp_inside():
    """Shue et al. magnetopause: point near Earth should be inside."""
    _recalc()
    # 3 RE sunward: clearly inside magnetopause
    xm, ym, zm, dist, id_ = geopack.shuetal_mgnp(5.0, -400.0, -5.0, 3.0, 0.0, 0.0)
    assert id_ == 1, f"Shue: expected inside (+1), got {id_}"


def test_shuetal_mgnp_outside():
    """Shue et al. magnetopause: 25 RE sunward should be outside."""
    _recalc()
    xm, ym, zm, dist, id_ = geopack.shuetal_mgnp(5.0, -400.0, -5.0, 25.0, 0.0, 0.0)
    assert id_ == -1, f"Shue: expected outside (-1), got {id_}"


def test_error_if_no_recalc():
    """Functions should raise RuntimeError when recalc() has not been called."""
    import geopack.geopack as _gp
    original = _gp._recalc_called
    _gp._recalc_called = False          # simulate un-initialised state
    try:
        with pytest.raises(RuntimeError, match="recalc"):
            geopack.igrf_gsw(4.0, 0.0, 0.0)
    finally:
        _gp._recalc_called = original   # restore


def test_trace_field_line():
    """trace_field_line should return a dict with expected keys and sensible values."""
    result = geopack.trace_field_line(
        T0, 4.0, 0.0, 0.0,
        kext='t89', maginput={'Kp': 3}, sysaxes=1,  # GEO input
    )
    assert 'Lm' in result
    assert 'Blocal' in result
    assert 'POSIT' in result
    assert 'Nposit' in result
    n = result['Nposit']
    # A closed field line near 4 RE should have at least a few dozen points
    assert n > 10, f"Suspiciously few trace points: {n}"
    # McIlwain L should be near 4 for input at 4 RE equatorial
    assert 1 < result['Lm'] < 20, f"Lm = {result['Lm']:.2f} out of plausible range"
    # POSIT shape: (npts, 3)
    assert result['POSIT'].shape == (n, 3)


# --------------------------------------------------------------------------- #
# trace() – external/internal field model variant tests
# --------------------------------------------------------------------------- #

# Standard driving parameters for trace model tests.
# Each entry: (exname, inname, iopt, parmod)
_TRACE_PARAMS = [
    # T89 (IOPT = Kp index 0-6) with both internal models
    ('t89',  'igrf', 3, None),
    ('t89',  'dip',  3, None),
    # T96 (PARMOD=[Pdyn, Dst, ByIMF, BzIMF, 0…]) with both internal models
    ('t96',  'igrf', 0, [2.0, -30.0,  2.0, -5.0, 0, 0, 0, 0, 0, 0]),
    ('t96',  'dip',  0, [2.0, -30.0,  2.0, -5.0, 0, 0, 0, 0, 0, 0]),
    # T01 (PARMOD=[Pdyn, Dst, ByIMF, BzIMF, G1, G2, 0…])
    ('t01',  'igrf', 0, [2.0, -30.0,  2.0, -5.0, 3.0, 1.0, 0, 0, 0, 0]),
    ('t01',  'dip',  0, [2.0, -30.0,  2.0, -5.0, 3.0, 1.0, 0, 0, 0, 0]),
    # T01S: parmod uses G2 and G3 (storm indices), not G1 and G2
    ('t01s', 'igrf', 0, [2.0, -30.0,  2.0, -5.0, 3.0, 1.0, 0, 0, 0, 0]),
    ('t01s', 'dip',  0, [2.0, -30.0,  2.0, -5.0, 3.0, 1.0, 0, 0, 0, 0]),
    # TS04 (PARMOD=[Pdyn, Dst, ByIMF, BzIMF, W1…W6])
    ('ts04', 'igrf', 0, [2.0, -30.0,  2.0, -5.0, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1]),
    ('ts04', 'dip',  0, [2.0, -30.0,  2.0, -5.0, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1]),
]
_TRACE_IDS = [f'{ex}/{inn}' for ex, inn, *_ in _TRACE_PARAMS]


@pytest.fixture(scope='module')
def ts07_available():
    """Return True when TS07D coefficient files are present.

    TS07D requires ~8.5 GB of external data configured via TS07_DATA_PATH.
    Tests that need this fixture are skipped automatically when files are absent.
    """
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    result = geopack.trace(-4.0, 0.0, 0.0, dir=-1.0, iopt=0,
                           exname='ts07', inname='igrf')
    xf, yf, zf = result['xf'], result['yf'], result['zf']
    try:
        ep_r = math.sqrt(xf**2 + yf**2 + zf**2)
        return result['npts'] < 3000 and abs(ep_r - 1.0) < 0.05
    except (ValueError, TypeError):
        return False


@pytest.mark.parametrize('exname,inname,iopt,parmod', _TRACE_PARAMS, ids=_TRACE_IDS)
def test_trace_starting_point_preserved(exname, inname, iopt, parmod):
    """trace() must store the starting coordinates as the first field-line point.

    This guards against the argument-count mismatch between RHAND_08's 9-arg
    EXNAME call and IRBEM's 7-arg field models, which previously caused Y and Z
    to be overwritten by the external field output before the first point was stored.
    """
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    xi, yi, zi = -4.0, 0.0, 0.0
    result = geopack.trace(xi, yi, zi, dir=-1.0, iopt=iopt, parmod=parmod,
                           exname=exname, inname=inname)
    assert result['npts'] > 0, f"{exname}/{inname}: trace returned no points"
    assert abs(result['xx'][0] - xi) < 1e-10, \
        f"{exname}/{inname}: xx[0]={result['xx'][0]:.6f}, expected {xi}"
    assert abs(result['yy'][0] - yi) < 1e-10, \
        f"{exname}/{inname}: yy[0]={result['yy'][0]:.6f}, expected {yi}"
    assert abs(result['zz'][0] - zi) < 1e-10, \
        f"{exname}/{inname}: zz[0]={result['zz'][0]:.6f}, expected {zi}"
    print(f"{exname}/{inname}: xf={result['xf']:.6f}, yf={result['yf']:.6f}, zf={result['zf']:.6f}")
    # Check the xf, yf, zf should be close to [-0.73, 0.00, 0.68] RE within 0.01 RE if inname is 'igrf' or close to [-0.67, 0.00, 0.74] RE if inname is 'dip'
    if inname == 'igrf':
        assert abs(result['xf'] + 0.73) < 0.01, f"{exname}/{inname}: xf={result['xf']:.6f}, expected -0.73"
        assert abs(result['yf']) < 0.01, f"{exname}/{inname}: yf={result['yf']:.6f}, expected 0.00"
        assert abs(result['zf'] - 0.68) < 0.01, f"{exname}/{inname}: zf={result['zf']:.6f}, expected 0.68"
    elif inname == 'dip':
        assert abs(result['xf'] + 0.67) < 0.01, f"{exname}/{inname}: xf={result['xf']:.6f}, expected -0.67"
        assert abs(result['yf']) < 0.01, f"{exname}/{inname}: yf={result['yf']:.6f}, expected 0.00"
        assert abs(result['zf'] - 0.74) < 0.01, f"{exname}/{inname}: zf={result['zf']:.6f}, expected 0.74"
    else:
        raise ValueError(f"Invalid inname: {inname}")

@pytest.mark.parametrize('exname,inname,iopt,parmod', _TRACE_PARAMS, ids=_TRACE_IDS)
def test_trace_endpoint_on_earth_surface(exname, inname, iopt, parmod):
    """Traced field-line footpoint must lie on the 1-RE inner boundary."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    result = geopack.trace(-4.0, 0.0, 0.0, dir=-1.0, iopt=iopt, parmod=parmod,
                           exname=exname, inname=inname)
    xf, yf, zf = result['xf'], result['yf'], result['zf']
    ep_r = math.sqrt(xf**2 + yf**2 + zf**2)
    assert abs(ep_r - 1.0) < 0.01, \
        f"{exname}/{inname}: footpoint |r|={ep_r:.6f} RE, expected 1.0"


def test_trace_ts07(ts07_available):
    """TS07D field-line trace should reach Earth's surface (skipped when data absent)."""
    if not ts07_available:
        pytest.skip('TS07D coefficient files not configured (set TS07_DATA_PATH)')
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    result = geopack.trace(-4.0, 0.0, 0.0, dir=-1.0, iopt=0,
                           exname='ts07', inname='igrf')
    assert abs(result['xx'][0] - (-4.0)) < 1e-10, "ts07: xx[0] mismatch"
    assert abs(result['yy'][0] - 0.0)   < 1e-10, "ts07: yy[0] mismatch"
    assert abs(result['zz'][0] - 0.0)   < 1e-10, "ts07: zz[0] mismatch"
    xf, yf, zf = result['xf'], result['yf'], result['zf']
    ep_r = math.sqrt(xf**2 + yf**2 + zf**2)
    assert abs(ep_r - 1.0) < 0.05, f"ts07 footpoint |r|={ep_r:.4f} RE"


# --------------------------------------------------------------------------- #
# IDL cross-validation tests
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def idl_ref():
    """Load IDL Geopack-2008 reference values from test_geopack.sav.

    The .sav file is produced by running test_geopack.pro in IDL:
        idl -e ".run test_geopack.pro"

    Tests that use this fixture are automatically skipped when the file is
    absent (CI-friendly: IDL is not required to run the main test suite).
    """
    sav_path = Path(__file__).parent / 'test_geopack.sav'
    if not sav_path.exists():
        pytest.skip('test_geopack.sav not found; run test_geopack.pro in IDL first')
    return sio.readsav(str(sav_path))


def test_recalc_vs_idl(idl_ref):
    """Dipole tilt from recalc() should match IDL geopack_recalc_08 within 1e-4 deg."""
    tilt = geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    idl_tilt = float(idl_ref['tilt'])
    assert abs(tilt - idl_tilt) < 1e-4, f"tilt: Python={tilt:.6f}, IDL={idl_tilt:.6f}"


def test_igrf_geo_vs_idl(idl_ref):
    """IGRF in geocentric spherical coords should match IDL reference within 1e-6 nT."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    br, bt, bp = geopack.igrf_geo(1.0, math.pi / 2, 0.0)
    assert abs(br - float(idl_ref['igrf_geo_br'])) < 1e-6, f"igrf_geo Br mismatch"
    assert abs(bt - float(idl_ref['igrf_geo_bt'])) < 1e-6, f"igrf_geo Bt mismatch"
    assert abs(bp - float(idl_ref['igrf_geo_bp'])) < 1e-6, f"igrf_geo Bp mismatch"


def test_igrf_gsw_vs_idl(idl_ref):
    """IGRF in GSW Cartesian coords should match IDL reference within 1e-6 nT."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    bx, by, bz = geopack.igrf_gsw(1.0, 0.0, 0.0)
    assert abs(bx - float(idl_ref['igrf_gsw_bx'])) < 1e-6, f"igrf_gsw Bx mismatch"
    assert abs(by - float(idl_ref['igrf_gsw_by'])) < 1e-6, f"igrf_gsw By mismatch"
    assert abs(bz - float(idl_ref['igrf_gsw_bz'])) < 1e-6, f"igrf_gsw Bz mismatch"


def test_dip_vs_idl(idl_ref):
    """Dipole field at (4, 0, 0) RE should match IDL reference within 1e-6 nT."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    bx, by, bz = geopack.dip(4.0, 0.0, 0.0)
    assert abs(bx - float(idl_ref['dip_bx'])) < 1e-6, f"dip Bx mismatch"
    assert abs(by - float(idl_ref['dip_by'])) < 1e-6, f"dip By mismatch"
    assert abs(bz - float(idl_ref['dip_bz'])) < 1e-6, f"dip Bz mismatch"


def test_sphcar_vs_idl(idl_ref):
    """sphcar() sphere→rect and rect→sphere should match IDL reference within 1e-10."""
    _recalc()
    x, y, z = geopack.sphcar(3.5, 1.2, 0.7, to_rect=True)
    assert abs(x - float(idl_ref['sphcar_x'])) < 1e-10, f"sphcar x mismatch"
    assert abs(y - float(idl_ref['sphcar_y'])) < 1e-10, f"sphcar y mismatch"
    assert abs(z - float(idl_ref['sphcar_z'])) < 1e-10, f"sphcar z mismatch"
    r, theta, phi = geopack.sphcar(x, y, z, to_rect=False)
    assert abs(r     - float(idl_ref['sphcar_r']))     < 1e-10, f"sphcar r roundtrip"
    assert abs(theta - float(idl_ref['sphcar_theta'])) < 1e-10, f"sphcar theta roundtrip"
    assert abs(phi   - float(idl_ref['sphcar_phi']))   < 1e-10, f"sphcar phi roundtrip"


def test_bspcar_vs_idl(idl_ref):
    """bspcar() should match IDL geopack_bspcar_08 within 1e-8 nT."""
    _recalc()
    bx, by, bz = geopack.bspcar(1.1, 0.5, 100.0, -200.0, 50.0)
    assert abs(bx - float(idl_ref['bspcar_bx'])) < 1e-8, f"bspcar Bx mismatch"
    assert abs(by - float(idl_ref['bspcar_by'])) < 1e-8, f"bspcar By mismatch"
    assert abs(bz - float(idl_ref['bspcar_bz'])) < 1e-8, f"bspcar Bz mismatch"


def test_bcarsp_vs_idl(idl_ref):
    """bcarsp() should match IDL geopack_bcarsp_08 within 1e-8 nT."""
    _recalc()
    theta, phi = 1.1, 0.5
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    bx = float(idl_ref['bspcar_bx'])
    by = float(idl_ref['bspcar_by'])
    bz = float(idl_ref['bspcar_bz'])
    br, bt, bp = geopack.bcarsp(x, y, z, bx, by, bz)
    assert abs(br - float(idl_ref['bcarsp_br'])) < 1e-8, f"bcarsp Br mismatch"
    assert abs(bt - float(idl_ref['bcarsp_bt'])) < 1e-8, f"bcarsp Bt mismatch"
    assert abs(bp - float(idl_ref['bcarsp_bp'])) < 1e-8, f"bcarsp Bp mismatch"


def test_conv_coord_geo_gsm_vs_idl(idl_ref):
    """GEO → GSW conversion should match IDL geopack_conv_coord_08 within 1e-8 RE."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    d1, d2, d3 = geopack.conv_coord(3.0, 1.5, -0.5, 'GEO', 'GSW')
    assert abs(d1 - float(idl_ref['cc_geo2gsm_d1'])) < 1e-8, f"GEO→GSW d1 mismatch"
    assert abs(d2 - float(idl_ref['cc_geo2gsm_d2'])) < 1e-8, f"GEO→GSW d2 mismatch"
    assert abs(d3 - float(idl_ref['cc_geo2gsm_d3'])) < 1e-8, f"GEO→GSW d3 mismatch"


def test_conv_coord_gsm_geo_vs_idl(idl_ref):
    """GSW → GEO roundtrip should match IDL reference within 1e-8 RE."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    d1 = float(idl_ref['cc_geo2gsm_d1'])
    d2 = float(idl_ref['cc_geo2gsm_d2'])
    d3 = float(idl_ref['cc_geo2gsm_d3'])
    x, y, z = geopack.conv_coord(d1, d2, d3, 'GSW', 'GEO')
    assert abs(x - float(idl_ref['cc_gsm2geo_d1'])) < 1e-8, f"GSW→GEO d1 mismatch"
    assert abs(y - float(idl_ref['cc_gsm2geo_d2'])) < 1e-8, f"GSW→GEO d2 mismatch"
    assert abs(z - float(idl_ref['cc_gsm2geo_d3'])) < 1e-8, f"GSW→GEO d3 mismatch"


def test_conv_coord_geo_gse_vs_idl(idl_ref):
    """GEO → GSE conversion should match IDL reference within 1e-8 RE."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    d1, d2, d3 = geopack.conv_coord(2.0, -1.0, 0.5, 'GEO', 'GSE')
    assert abs(d1 - float(idl_ref['cc_geo2gse_d1'])) < 1e-8, f"GEO→GSE d1 mismatch"
    assert abs(d2 - float(idl_ref['cc_geo2gse_d2'])) < 1e-8, f"GEO→GSE d2 mismatch"
    assert abs(d3 - float(idl_ref['cc_geo2gse_d3'])) < 1e-8, f"GEO→GSE d3 mismatch"


def test_t89_vs_idl(idl_ref):
    """T89 external field at (-4, 2, 1) RE should match IDL reference within 1e-4 nT."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    bx, by, bz = geopack.t89(3, -4.0, 2.0, 1.0)
    assert abs(bx - float(idl_ref['t89_bx'])) < 1.5e-2, f"T89 Bx mismatch"
    assert abs(by - float(idl_ref['t89_by'])) < 1.5e-2, f"T89 By mismatch"
    assert abs(bz - float(idl_ref['t89_bz'])) < 1.5e-2, f"T89 Bz mismatch"


def test_t96_vs_idl(idl_ref):
    """T96 external field at (-4, 2, 1) RE should match IDL reference within 1e-4 nT."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    parmod = [2.0, -10.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0]
    bx, by, bz = geopack.t96(parmod, -4.0, 2.0, 1.0)
    assert abs(bx - float(idl_ref['t96_bx'])) < 2.5e-2, f"T96 Bx mismatch"
    assert abs(by - float(idl_ref['t96_by'])) < 2.5e-2, f"T96 By mismatch"
    assert abs(bz - float(idl_ref['t96_bz'])) < 2.5e-2, f"T96 Bz mismatch"


def test_t01_vs_idl(idl_ref):
    """T01 external field at (-4, 2, 1) RE should match IDL reference within 1e-4 nT."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    parmod = [2.0, -10.0, 0.0, -5.0, 3.0, 1.0, 0, 0, 0, 0]
    bx, by, bz = geopack.t01(parmod, -4.0, 2.0, 1.0)
    assert abs(bx - float(idl_ref['t01_bx'])) < 1.5e-2, f"T01 Bx mismatch"
    assert abs(by - float(idl_ref['t01_by'])) < 1.5e-2, f"T01 By mismatch"
    assert abs(bz - float(idl_ref['t01_bz'])) < 1.5e-2, f"T01 Bz mismatch"


def test_t01s_vs_idl(idl_ref):
    """T01_S storm-time external field at (-4, 2, 1) RE should match IDL reference within 1.5e-2 nT."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    parmod = [4.0, -100.0, 2.0, -8.0, 6.0, 3.0, 0, 0, 0, 0]
    bx, by, bz = geopack.t01s(parmod, -4.0, 2.0, 1.0)
    assert abs(bx - float(idl_ref['t01s_bx'])) < 4e-2, f"T01S Bx mismatch"
    assert abs(by - float(idl_ref['t01s_by'])) < 4e-2, f"T01S By mismatch"
    assert abs(bz - float(idl_ref['t01s_bz'])) < 4e-2, f"T01S Bz mismatch"


def test_ts04_vs_idl(idl_ref):
    """TS04 external field at (-4, 2, 1) RE should match IDL reference within 1e-4 nT."""
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    parmod = [2.0, -10.0, 0.0, -5.0, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1]
    bx, by, bz = geopack.ts04(parmod, -4.0, 2.0, 1.0)
    assert abs(bx - float(idl_ref['ts04_bx'])) < 1.5e-2, f"TS04 Bx mismatch"
    assert abs(by - float(idl_ref['ts04_by'])) < 1.5e-2, f"TS04 By mismatch"
    assert abs(bz - float(idl_ref['ts04_bz'])) < 1.5e-2, f"TS04 Bz mismatch"


def test_t96_mgnp_vs_idl(idl_ref):
    """T96 magnetopause results should match IDL for both inside and outside points."""
    _recalc()
    # Inside: (3, 0, 0) RE
    xm, ym, zm, dist, id_ = geopack.t96_mgnp(5.0, -400.0, 3.0, 0.0, 0.0)
    assert id_ == int(idl_ref['t96mgnp_in_id']), f"T96 mgnp inside id mismatch"
    assert abs(xm   - float(idl_ref['t96mgnp_in_xm']))   < 1e-6
    assert abs(ym   - float(idl_ref['t96mgnp_in_ym']))   < 1e-6
    assert abs(zm   - float(idl_ref['t96mgnp_in_zm']))   < 1e-6
    assert abs(dist - float(idl_ref['t96mgnp_in_dist'])) < 1e-6
    # Outside: (25, 0, 0) RE
    xm, ym, zm, dist, id_ = geopack.t96_mgnp(5.0, -400.0, 25.0, 0.0, 0.0)
    assert id_ == int(idl_ref['t96mgnp_out_id']), f"T96 mgnp outside id mismatch"
    assert abs(xm   - float(idl_ref['t96mgnp_out_xm']))   < 1e-6
    assert abs(ym   - float(idl_ref['t96mgnp_out_ym']))   < 1e-6
    assert abs(zm   - float(idl_ref['t96mgnp_out_zm']))   < 1e-6
    assert abs(dist - float(idl_ref['t96mgnp_out_dist'])) < 1e-6


def test_shuetal_mgnp_vs_idl(idl_ref):
    """Shue et al. magnetopause results should match IDL for inside and outside points."""
    _recalc()
    # Inside: (3, 0, 0) RE
    xm, ym, zm, dist, id_ = geopack.shuetal_mgnp(5.0, -400.0, -5.0, 3.0, 0.0, 0.0)
    assert id_ == int(idl_ref['shue_in_id']), f"Shue mgnp inside id mismatch"
    assert abs(xm   - float(idl_ref['shue_in_xm']))   < 1e-6
    assert abs(ym   - float(idl_ref['shue_in_ym']))   < 1e-6
    assert abs(zm   - float(idl_ref['shue_in_zm']))   < 1e-6
    assert abs(dist - float(idl_ref['shue_in_dist'])) < 1e-6
    # Outside: (25, 0, 0) RE
    xm, ym, zm, dist, id_ = geopack.shuetal_mgnp(5.0, -400.0, -5.0, 25.0, 0.0, 0.0)
    assert id_ == int(idl_ref['shue_out_id']), f"Shue mgnp outside id mismatch"
    assert abs(xm   - float(idl_ref['shue_out_xm']))   < 1e-6
    assert abs(ym   - float(idl_ref['shue_out_ym']))   < 1e-6
    assert abs(zm   - float(idl_ref['shue_out_zm']))   < 1e-6
    assert abs(dist - float(idl_ref['shue_out_dist'])) < 1e-6


def test_trace_vs_idl(idl_ref):
    """Field-line trace endpoint from (4, 0, 0) RE should match IDL within 0.01 RE.

    IDL geopack_trace_08 dir=-1 (parallel to B) corresponds to Python trace() dir=+1.
    """
    geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
    result = geopack.trace(-4.0, 0.0, 0.0, dir=-1.0, iopt=3, exname='t89', inname='igrf')
    assert abs(result['xf'] - float(idl_ref['trace_xf'])) < 0.01, f"trace xf mismatch"
    assert abs(result['yf'] - float(idl_ref['trace_yf'])) < 0.01, f"trace yf mismatch"
    assert abs(result['zf'] - float(idl_ref['trace_zf'])) < 0.01, f"trace zf mismatch"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    tests = [
        test_recalc,
        test_igrf_geo,
        test_igrf_gsw,
        test_dip,
        test_sphcar_roundtrip,
        test_bspcar_bcarsp_roundtrip,
        test_conv_coord_identity,
        test_conv_coord_geo_gsm_roundtrip,
        test_conv_coord_geo_gse_roundtrip,
        test_conv_coord_preserves_length,
        test_t89,
        test_t96,
        test_t96_mgnp_inside,
        test_t96_mgnp_outside,
        test_shuetal_mgnp_inside,
        test_shuetal_mgnp_outside,
        test_error_if_no_recalc,
        test_trace_field_line,
        *[functools.partial(test_trace_starting_point_preserved, *p)
          for p in _TRACE_PARAMS],
        *[functools.partial(test_trace_endpoint_on_earth_surface, *p)
          for p in _TRACE_PARAMS],
    ]

    # Load IDL reference data if available and append cross-validation tests
    _sav_path = Path(__file__).parent / 'test_geopack.sav'
    try:
        _idl_ref = sio.readsav(str(_sav_path))
        _idl_tests = [
            test_recalc_vs_idl,
            test_igrf_geo_vs_idl,
            test_igrf_gsw_vs_idl,
            test_dip_vs_idl,
            test_sphcar_vs_idl,
            test_bspcar_vs_idl,
            test_bcarsp_vs_idl,
            test_conv_coord_geo_gsm_vs_idl,
            test_conv_coord_gsm_geo_vs_idl,
            test_conv_coord_geo_gse_vs_idl,
            test_t89_vs_idl,
            test_t96_vs_idl,
            test_t01_vs_idl,
            test_t01s_vs_idl,
            test_ts04_vs_idl,
            test_t96_mgnp_vs_idl,
            test_shuetal_mgnp_vs_idl,
            test_trace_vs_idl,
        ]
        tests += [functools.partial(fn, _idl_ref) for fn in _idl_tests]
    except FileNotFoundError:
        print(f"  SKIP  IDL cross-validation tests (test_geopack.sav not found)")
    except ImportError:
        print(f"  SKIP  IDL cross-validation tests (scipy not installed)")

    passed = 0
    failed = 0
    for fn in tests:
        name = getattr(fn, '__name__', None) or fn.func.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
