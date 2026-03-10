"""
Tutorial: Python geopack wrapper for Geopack-2008 and Tsyganenko field models.

This script demonstrates the key functions available in the geopack module,
covering:
  - Initialisation (recalc)
  - Internal field models (IGRF, dipole)
  - Coordinate conversion utilities
  - External field models (T89, T96, T01, T01S, TS04, TS07D)
  - Magnetopause models (T96, Shue et al.)
  - Low-level field-line tracing (trace)
  - High-level field-line tracing (trace_field_line)

Run with:
    python python/geopack/tutorial_geopack.py
"""

import math
import os
import sys

import numpy as np

# Allow running directly from the repo root or from the geopack/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geopack

# Reference epoch used throughout this tutorial
T0 = np.datetime64('2000-01-11T17:00:00')


# ===========================================================================
# 1. SETUP: recalc()
# ===========================================================================
print("=" * 60)
print("1. SETUP: recalc()")
print("=" * 60)

# Default call – solar wind assumed purely antisunward at -400 km/s
tilt_default = geopack.recalc(T0)
print(f"  recalc(T0) – dipole tilt (GSW/GSM) = {tilt_default:.4f} deg")

# Explicit solar wind vector (vgsex/vgsey/vgsez in km/s, GSE frame)
tilt_sw = geopack.recalc(T0, vgsex=-450.0, vgsey=5.0, vgsez=-2.0)
print(f"  recalc(T0, vgsex=-450, vgsey=5, vgsez=-2) – tilt = {tilt_sw:.4f} deg")
print(f"  (Tilt changes slightly because GSW frame shifts from GSM with non-zero By/Bz.)")

# Reset to the simple default for the rest of the tutorial
geopack.recalc(T0)


# ===========================================================================
# 2. INTERNAL FIELD MODELS
# ===========================================================================
print()
print("=" * 60)
print("2. INTERNAL FIELD MODELS")
print("=" * 60)

# --- igrf_geo: IGRF in geocentric spherical coordinates ---
print("\n  igrf_geo(r, theta, phi)")
print("  Inputs: r=1.0 RE (surface), theta=pi/2 (equator), phi=0 (Greenwich)")
br, bt, bp = geopack.igrf_geo(1.0, math.pi / 2, 0.0)
bmag = math.sqrt(br**2 + bt**2 + bp**2)
print(f"    Br={br:.2f} nT, Btheta={bt:.2f} nT, Bphi={bp:.2f} nT")
print(f"    |B| = {bmag:.2f} nT  (expected ~25 000–65 000 nT at Earth's surface)")

# --- igrf_gsw: IGRF in GSW Cartesian coordinates ---
print("\n  igrf_gsw(x, y, z)")
print("  Inputs: position (1.0, 0.0, 0.0) RE in GSW")
bx, by, bz = geopack.igrf_gsw(1.0, 0.0, 0.0)
bmag = math.sqrt(bx**2 + by**2 + bz**2)
print(f"    Bx={bx:.2f} nT, By={by:.2f} nT, Bz={bz:.2f} nT")
print(f"    |B| = {bmag:.2f} nT")

# --- dip: Pure dipole field in GSW Cartesian coordinates ---
print("\n  dip(x, y, z)")
print("  Inputs: position (4.0, 0.0, 0.0) RE in GSW")
bx, by, bz = geopack.dip(4.0, 0.0, 0.0)
bmag = math.sqrt(bx**2 + by**2 + bz**2)
print(f"    Bx={bx:.4f} nT, By={by:.4f} nT, Bz={bz:.4f} nT")
print(f"    |B| = {bmag:.4f} nT  (expected ~470 nT at 4 RE; scales as 1/r^3)")


# ===========================================================================
# 3. COORDINATE CONVERSION UTILITIES
# ===========================================================================
print()
print("=" * 60)
print("3. COORDINATE CONVERSION UTILITIES")
print("=" * 60)

# --- sphcar: spherical <-> Cartesian ---
print("\n  sphcar – spherical <-> Cartesian")
r0, theta0, phi0 = 3.5, 1.2, 0.7
x, y, z = geopack.sphcar(r0, theta0, phi0, to_rect=True)
print(f"  sphere->rect: (r={r0}, theta={theta0}, phi={phi0})")
print(f"    -> (x={x:.6f}, y={y:.6f}, z={z:.6f}) RE")

r1, theta1, phi1 = geopack.sphcar(x, y, z, to_rect=False)
print(f"  rect->sphere (round-trip): (x={x:.6f}, y={y:.6f}, z={z:.6f})")
print(f"    -> (r={r1:.6f}, theta={theta1:.6f}, phi={phi1:.6f})")
print(f"  Round-trip errors: dr={abs(r1-r0):.2e}, dtheta={abs(theta1-theta0):.2e}, dphi={abs(phi1-phi0):.2e}")

# --- bspcar / bcarsp: field vector spherical <-> Cartesian ---
print("\n  bspcar / bcarsp – field vector conversion")
theta, phi = 1.1, 0.5
br0, bt0, bp0 = 100.0, -200.0, 50.0
bx, by, bz = geopack.bspcar(theta, phi, br0, bt0, bp0)
print(f"  bspcar: (theta={theta}, phi={phi}), B_sph=({br0}, {bt0}, {bp0}) nT")
print(f"    -> B_cart = ({bx:.6f}, {by:.6f}, {bz:.6f}) nT")

# For bcarsp the position in Cartesian is needed (unit vector on the sphere)
x_pos = math.sin(theta) * math.cos(phi)
y_pos = math.sin(theta) * math.sin(phi)
z_pos = math.cos(theta)
br1, bt1, bp1 = geopack.bcarsp(x_pos, y_pos, z_pos, bx, by, bz)
print(f"  bcarsp (round-trip): -> B_sph = ({br1:.6f}, {bt1:.6f}, {bp1:.6f}) nT")
print(f"  Round-trip errors: dBr={abs(br1-br0):.2e}, dBt={abs(bt1-bt0):.2e}, dBp={abs(bp1-bp0):.2e}")

# --- conv_coord: coordinate system conversions ---
print("\n  conv_coord – coordinate system conversions")
x0, y0, z0 = 3.0, 1.5, -0.5

# GEO -> GSM round-trip
xg, yg, zg = geopack.conv_coord(x0, y0, z0, 'GEO', 'GSM')
x1, y1, z1 = geopack.conv_coord(xg, yg, zg, 'GSM', 'GEO')
print(f"  GEO -> GSM: ({x0}, {y0}, {z0}) -> ({xg:.6f}, {yg:.6f}, {zg:.6f}) RE")
print(f"  GSM -> GEO (round-trip): -> ({x1:.6f}, {y1:.6f}, {z1:.6f}) RE")

# GEO -> multiple systems, verify magnitude preservation
x0, y0, z0 = 4.0, 2.0, -1.0
r0 = math.sqrt(x0**2 + y0**2 + z0**2)
print(f"\n  Magnitude-preserving rotations from GEO ({x0}, {y0}, {z0}) RE, |r|={r0:.4f} RE:")
for dst in ('MAG', 'SM', 'GSM', 'GSE', 'GEI'):
    xd, yd, zd = geopack.conv_coord(x0, y0, z0, 'GEO', dst)
    rd = math.sqrt(xd**2 + yd**2 + zd**2)
    print(f"    GEO->{dst}: ({xd:.4f}, {yd:.4f}, {zd:.4f}), |r|={rd:.6f} RE")


# ===========================================================================
# 4. EXTERNAL FIELD MODELS
# ===========================================================================
print()
print("=" * 60)
print("4. EXTERNAL FIELD MODELS")
print("=" * 60)
print("  All external fields evaluated at (-4, 2, 1) RE in GSW.")

geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
xp, yp, zp = -4.0, 2.0, 1.0

# --- T89 ---
print("\n  t89(iopt, x, y, z)  – Tsyganenko 1989")
print("  iopt: 1=Kp<1, 2=Kp1-2, 3=Kp2-3, 4=Kp3-4, 5=Kp4-5, 6=Kp5-6, 7=Kp>=6")
bx, by, bz = geopack.t89(3, xp, yp, zp)
bmag = math.sqrt(bx**2 + by**2 + bz**2)
print(f"    iopt=3 (Kp 2-3): Bx={bx:.4f}, By={by:.4f}, Bz={bz:.4f} nT, |B|={bmag:.4f} nT")

# --- T96 ---
print("\n  t96(parmod, x, y, z)  – Tsyganenko 1996")
print("  parmod: [Pdyn(nPa), Dst(nT), ByIMF(nT), BzIMF(nT), 0, 0, 0, 0, 0, 0]")
parmod_t96 = [2.0, -10.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0]
bx, by, bz = geopack.t96(parmod_t96, xp, yp, zp)
bmag = math.sqrt(bx**2 + by**2 + bz**2)
print(f"    Pdyn=2 nPa, Dst=-10 nT, Bz=-5 nT: Bx={bx:.4f}, By={by:.4f}, Bz={bz:.4f} nT, |B|={bmag:.4f} nT")

# --- T01 ---
print("\n  t01(parmod, x, y, z)  – Tsyganenko 2001")
print("  parmod: [Pdyn, Dst, ByIMF, BzIMF, G1, G2, 0, 0, 0, 0]")
parmod_t01 = [2.0, -10.0, 0.0, -5.0, 3.0, 1.0, 0, 0, 0, 0]
bx, by, bz = geopack.t01(parmod_t01, xp, yp, zp)
bmag = math.sqrt(bx**2 + by**2 + bz**2)
print(f"    G1=3.0, G2=1.0: Bx={bx:.4f}, By={by:.4f}, Bz={bz:.4f} nT, |B|={bmag:.4f} nT")

# --- T01S ---
print("\n  t01s(parmod, x, y, z)  – Tsyganenko 2001 storm-time")
print("  parmod: [Pdyn, Dst, ByIMF, BzIMF, G2, G3, 0, 0, 0, 0]  (storm indices)")
geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)
parmod_t01s = [4.0, -100.0, 2.0, -8.0, 6.0, 3.0, 0, 0, 0, 0]
bx, by, bz = geopack.t01s(parmod_t01s, xp, yp, zp)
bmag = math.sqrt(bx**2 + by**2 + bz**2)
print(f"    Bx={bx:.4f}, By={by:.4f}, Bz={bz:.4f} nT, |B|={bmag:.4f} nT")

# --- TS04 ---
print("\n  ts04(parmod, x, y, z)  – Tsyganenko-Sitnov 2004")
print("  parmod: [Pdyn, Dst, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6]")
parmod_ts04 = [2.0, -10.0, 0.0, -5.0, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1]
bx, by, bz = geopack.ts04(parmod_ts04, xp, yp, zp)
bmag = math.sqrt(bx**2 + by**2 + bz**2)
print(f"    W1-W6 provided: Bx={bx:.4f}, By={by:.4f}, Bz={bz:.4f} nT, |B|={bmag:.4f} nT")

# --- TS07 (optional – requires external data files) ---
print("\n  ts07(x, y, z)  – Tsyganenko-Sitnov 2007 (requires TS07_DATA_PATH)")
try:
    bx, by, bz = geopack.ts07(xp, yp, zp)
    if math.isnan(bx):
        print(f"    NOTE: TS07D coefficient files not loaded (returned NaN).")
        print(f"    To enable: set TS07_DATA_PATH and run setup_ts07d_files.sh")
    else:
        bmag = math.sqrt(bx**2 + by**2 + bz**2)
        print(f"    Bx={bx:.4f}, By={by:.4f}, Bz={bz:.4f} nT, |B|={bmag:.4f} nT")
except Exception as e:
    print(f"    NOTE: TS07D not available ({e})")
    print(f"    To enable: set TS07_DATA_PATH and run setup_ts07d_files.sh")

# --- Combined field = internal + external ---
print("\n  Combined field: IGRF + T89 at (4, 0, 0) RE")
geopack.recalc(T0)
bxi, byi, bzi = geopack.igrf_gsw(4.0, 0.0, 0.0)
bxe, bye, bze = geopack.t89(3, 4.0, 0.0, 0.0)
bx_tot = bxi + bxe
by_tot = byi + bye
bz_tot = bzi + bze
bmag_tot = math.sqrt(bx_tot**2 + by_tot**2 + bz_tot**2)
print(f"    IGRF:   Bx={bxi:.4f}, By={byi:.4f}, Bz={bzi:.4f} nT")
print(f"    T89:    Bx={bxe:.4f}, By={bye:.4f}, Bz={bze:.4f} nT")
print(f"    Total:  Bx={bx_tot:.4f}, By={by_tot:.4f}, Bz={bz_tot:.4f} nT, |B|={bmag_tot:.4f} nT")


# ===========================================================================
# 5. MAGNETOPAUSE MODELS
# ===========================================================================
print()
print("=" * 60)
print("5. MAGNETOPAUSE MODELS")
print("=" * 60)
print("  Returns (x_mgnp, y_mgnp, z_mgnp, dist, id)")
print("  id = +1 inside magnetopause, -1 outside")

geopack.recalc(T0)

# --- T96 magnetopause ---
print("\n  t96_mgnp(xn_pd, vel, x, y, z)")
print("  xn_pd=5.0 cm^-3 (proton density), vel=-400 km/s (solar wind speed)")
xm, ym, zm, dist, id_ = geopack.t96_mgnp(5.0, -400.0, 3.0, 0.0, 0.0)
print(f"  Point (3, 0, 0) RE (near Earth):")
print(f"    Nearest MP point: ({xm:.4f}, {ym:.4f}, {zm:.4f}) RE, dist={dist:.4f} RE, id={id_} ({'inside' if id_==1 else 'outside'})")

xm, ym, zm, dist, id_ = geopack.t96_mgnp(5.0, -400.0, 25.0, 0.0, 0.0)
print(f"  Point (25, 0, 0) RE (beyond subsolar nose ~10 RE):")
print(f"    Nearest MP point: ({xm:.4f}, {ym:.4f}, {zm:.4f}) RE, dist={dist:.4f} RE, id={id_} ({'inside' if id_==1 else 'outside'})")

# --- Shue et al. magnetopause ---
print("\n  shuetal_mgnp(xn_pd, vel, bzimf, x, y, z)")
print("  Additional param: bzimf=-5.0 nT (IMF Bz)")
xm, ym, zm, dist, id_ = geopack.shuetal_mgnp(5.0, -400.0, -5.0, 3.0, 0.0, 0.0)
print(f"  Point (3, 0, 0) RE:")
print(f"    Nearest MP point: ({xm:.4f}, {ym:.4f}, {zm:.4f}) RE, dist={dist:.4f} RE, id={id_} ({'inside' if id_==1 else 'outside'})")

xm, ym, zm, dist, id_ = geopack.shuetal_mgnp(5.0, -400.0, -5.0, 25.0, 0.0, 0.0)
print(f"  Point (25, 0, 0) RE:")
print(f"    Nearest MP point: ({xm:.4f}, {ym:.4f}, {zm:.4f}) RE, dist={dist:.4f} RE, id={id_} ({'inside' if id_==1 else 'outside'})")


# ===========================================================================
# 6. LOW-LEVEL FIELD LINE TRACING: trace()
# ===========================================================================
print()
print("=" * 60)
print("6. LOW-LEVEL FIELD LINE TRACING: trace()")
print("=" * 60)
print("  Traces along B using Geopack TRACE_08 in GSW coordinates.")
print("  Returns dict with xf/yf/zf (endpoint), xx/yy/zz (path arrays), npts.")

geopack.recalc(T0, vgsex=-400.0, vgsey=0.0, vgsez=0.0)

# Trace from (-4, 0, 0) RE using T89 + IGRF, tracing antiparallel to B
xi, yi, zi = -4.0, 0.0, 0.0
result = geopack.trace(xi, yi, zi, dir=-1.0, iopt=3, exname='t89', inname='igrf')

print(f"\n  Starting point: ({xi}, {yi}, {zi}) RE")
print(f"  Model: T89 (iopt=3, Kp 2-3) + IGRF internal, dir=-1 (antiparallel to B)")
print(f"  Number of trace points: {result['npts']}")
print(f"  Endpoint (footpoint): ({result['xf']:.4f}, {result['yf']:.4f}, {result['zf']:.4f}) RE")
ep_r = math.sqrt(result['xf']**2 + result['yf']**2 + result['zf']**2)
print(f"  |endpoint| = {ep_r:.4f} RE  (should be ~1.0, i.e. Earth surface)")
print(f"  First point in path: ({result['xx'][0]:.6f}, {result['yy'][0]:.6f}, {result['zz'][0]:.6f}) RE")
print(f"  Last point in path:  ({result['xx'][-1]:.6f}, {result['yy'][-1]:.6f}, {result['zz'][-1]:.6f}) RE")
print(f"  xx array shape: {result['xx'].shape}, yy: {result['yy'].shape}, zz: {result['zz'].shape}")

# Show range of the traced field line
print(f"  X range: [{result['xx'].min():.3f}, {result['xx'].max():.3f}] RE")
print(f"  Y range: [{result['yy'].min():.3f}, {result['yy'].max():.3f}] RE")
print(f"  Z range: [{result['zz'].min():.3f}, {result['zz'].max():.3f}] RE")


# ===========================================================================
# 7. HIGH-LEVEL FIELD LINE TRACING: trace_field_line()
# ===========================================================================
print()
print("=" * 60)
print("7. HIGH-LEVEL FIELD LINE TRACING: trace_field_line()")
print("=" * 60)
print("  Uses IRBEM trace_field_line1 – handles coordinate conversion internally.")
print("  Does NOT require a prior recalc() call.")
print("  Returns: Lm, Blocal, Bmin, XJ, POSIT (GEO coords), Nposit.")

result = geopack.trace_field_line(
    T0, 4.0, 0.0, 0.0,
    kext='t89',
    maginput={'Kp': 3},
    sysaxes=1,   # 1 = GEO input
)

print(f"\n  Input: (4.0, 0.0, 0.0) RE in GEO, T89 with Kp=3, epoch={T0}")
print(f"  McIlwain L (Lm):      {result['Lm']:.4f}")
print(f"  Bmin along line:      {result['Bmin']:.4f} nT")
print(f"  Second invariant XJ:  {result['XJ']:.4f} RE")
print(f"  Number of points:     {result['Nposit']}")
print(f"  POSIT shape:          {result['POSIT'].shape}  (Nposit x 3, GEO coordinates)")
print(f"  Blocal shape:         {result['Blocal'].shape}")
if result['Nposit'] > 0:
    print(f"  First POSIT point: ({result['POSIT'][0,0]:.4f}, {result['POSIT'][0,1]:.4f}, {result['POSIT'][0,2]:.4f}) RE GEO")
    print(f"  Last  POSIT point: ({result['POSIT'][-1,0]:.4f}, {result['POSIT'][-1,1]:.4f}, {result['POSIT'][-1,2]:.4f}) RE GEO")
    print(f"  Blocal at equator (min): {result['Blocal'].min():.4f} nT")
    print(f"  Blocal at footpoints (max): {result['Blocal'].max():.4f} nT")


print()
print("=" * 60)
print("Tutorial complete.")
print("=" * 60)
