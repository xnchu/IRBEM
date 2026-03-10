pro test_geopack

    ; test_geopack.pro
    ;
    ; IDL Geopack-2008 reference values for cross-validation with Python geopack.py.
    ;
    ; Requires the IDL Geopack DLM (Haje Korth, JHU/APL):
    ;   https://ampere.jhuapl.edu/code/idl_geopack.html
    ;
    ; Run with:
    ;   idl -e ".run test_geopack.pro"
    ; or interactively:
    ;   IDL> .run test_geopack.pro
    ;
    ; Output: test_geopack.sav (written in the same directory as this script)

    on_error, 2   ; return to caller on error (surfaces line numbers)

    ; ---- Test epoch: 2000-01-11T17:00:00 UT (year=2000, doy=11) ----
    iyear   = 2000
    idoy    = 11
    ihour   = 17
    iminute = 0
    isecond = 0
    vgse    = [-400.0d, 0.0d, 0.0d]

    print, 'Computing Geopack-2008 reference values...'

    ; 1. recalc_08 – compute internal rotation matrices and retrieve dipole tilt
    ;    vgse keyword: solar wind velocity in GSE (km/s); [-400,0,0] = purely antisunward
    ;    tilt keyword: output dipole tilt angle (degrees) in the GSW frame
    print, '  1. geopack_recalc_08'
    geopack_recalc_08, iyear, idoy, ihour, iminute, isecond, vgse=vgse, tilt=tilt

    ; 2. IGRF in geocentric spherical coordinates (r=1 RE, theta=pi/2, phi=0)
    print, '  2. geopack_igrf_geo_08'
    geopack_igrf_geo_08, 1.0d, !dpi/2.0d, 0.0d, igrf_geo_br, igrf_geo_bt, igrf_geo_bp

    ; 3. IGRF in GSW Cartesian coordinates at (1, 0, 0) RE
    print, '  3. geopack_igrf_gsw_08'
    geopack_igrf_gsw_08, 1.0d, 0.0d, 0.0d, igrf_gsw_bx, igrf_gsw_by, igrf_gsw_bz

    ; 4. Pure dipole field in GSW Cartesian coordinates at (4, 0, 0) RE
    print, '  4. geopack_dip_08'
    geopack_dip_08, 4.0d, 0.0d, 0.0d, dip_bx, dip_by, dip_bz

    ; 5. Spherical -> Cartesian: (r, theta, phi) = (3.5, 1.2, 0.7)
    print, '  5. geopack_sphcar_08 (to_rect)'
    geopack_sphcar_08, 3.5d, 1.2d, 0.7d, sphcar_x, sphcar_y, sphcar_z, /to_rect

    ; 6. Cartesian -> Spherical: roundtrip from the result above
    print, '  6. geopack_sphcar_08 (to_sphere)'
    geopack_sphcar_08, sphcar_x, sphcar_y, sphcar_z, sphcar_r, sphcar_theta, sphcar_phi, /to_sphere

    ; 7. bspcar_08: spherical field -> Cartesian field
    ;    At (theta, phi) = (1.1, 0.5), field (br, bt, bp) = (100, -200, 50) nT
    print, '  7. geopack_bspcar_08'
    theta_bs = 1.1d
    phi_bs   = 0.5d
    geopack_bspcar_08, theta_bs, phi_bs, 100.0d, -200.0d, 50.0d, bspcar_bx, bspcar_by, bspcar_bz

    ; 8. bcarsp_08: Cartesian field -> spherical field (inverse of step 7)
    ;    Unit-vector position on the sphere at (theta_bs, phi_bs)
    print, '  8. geopack_bcarsp_08'
    x_bs = sin(theta_bs) * cos(phi_bs)
    y_bs = sin(theta_bs) * sin(phi_bs)
    z_bs = cos(theta_bs)
    geopack_bcarsp_08, x_bs, y_bs, z_bs, bspcar_bx, bspcar_by, bspcar_bz, $
        bcarsp_br, bcarsp_bt, bcarsp_bp

    ; 9. Coordinate conversion: GEO -> GSW (=GSM when vgsey=vgsez=0) at (3, 1.5, -0.5) RE
    print, '  9. geopack_conv_coord_08 (GEO->GSW)'
    geopack_conv_coord_08, 3.0d, 1.5d, -0.5d, cc_geo2gsm_d1, cc_geo2gsm_d2, cc_geo2gsm_d3, $
        /from_geo, /to_gsw

    ; 10. Coordinate conversion: GSW -> GEO (roundtrip from step 9)
    print, '  10. geopack_conv_coord_08 (GSW->GEO)'
    geopack_conv_coord_08, cc_geo2gsm_d1, cc_geo2gsm_d2, cc_geo2gsm_d3, $
        cc_gsm2geo_d1, cc_gsm2geo_d2, cc_gsm2geo_d3, /from_gsw, /to_geo

    ; 11. Coordinate conversion: GEO -> GSE at (2, -1, 0.5) RE
    print, '  11. geopack_conv_coord_08 (GEO->GSE)'
    geopack_conv_coord_08, 2.0d, -1.0d, 0.5d, cc_geo2gse_d1, cc_geo2gse_d2, cc_geo2gse_d3, $
        /from_geo, /to_gse

    ; 12. T89 external field at (-4, 2, 1) RE, iopt=3
    print, '  12. geopack_t89'
    geopack_t89, 3, -4.0d, 2.0d, 1.0d, t89_bx, t89_by, t89_bz

    ; 13. T96 external field at (-4, 2, 1) RE
    ;    parmod: [Pdyn(nPa), Dst(nT), ByIMF(nT), BzIMF(nT), 0, 0, 0, 0, 0, 0]
    print, '  13. geopack_t96'
    parmod_t96 = [2.0d, -10.0d, 0.0d, -5.0d, 0.0d, 0.0d, 0.0d, 0.0d, 0.0d, 0.0d]
    geopack_t96, parmod_t96, -4.0d, 2.0d, 1.0d, t96_bx, t96_by, t96_bz

    ; 14. T01 external field at (-4, 2, 1) RE
    ;    parmod: [Pdyn, Dst, ByIMF, BzIMF, G1, G2, 0, 0, 0, 0]
    print, '  14. geopack_t01'
    parmod_t01 = [2.0d, -10.0d, 0.0d, -5.0d, 3.0d, 1.0d, 0.0d, 0.0d, 0.0d, 0.0d]
    geopack_t01, parmod_t01, -4.0d, 2.0d, 1.0d, t01_bx, t01_by, t01_bz

    ; 15. TS04 external field at (-4, 2, 1) RE
    ;    parmod: [Pdyn, Dst, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6]
    print, '  15. geopack_ts04'
    parmod_ts04 = [2.0d, -10.0d, 0.0d, -5.0d, 0.1d, 0.2d, 0.1d, 0.3d, 0.2d, 0.1d]
    geopack_ts04, parmod_ts04, -4.0d, 2.0d, 1.0d, ts04_bx, ts04_by, ts04_bz

    ; 21. T01_S storm-time external field at (-4, 2, 1) RE
    ;    parmod: [Pdyn(nPa), Dst(nT), ByIMF(nT), BzIMF(nT), G2, G3, 0, 0, 0, 0]
    ;    Storm conditions: Dst=-100 nT, elevated G2/G3 storm-time indices
    ;    (SEE TSYGANENKO, SINGER, AND KASPER [2003] FOR EXACT DEFINITIONS OF G2 AND G3)
    print, '  21. geopack_t01s'
    parmod_t01s = [4.0d, -100.0d, 2.0d, -8.0d, 6.0d, 3.0d, 0.0d, 0.0d, 0.0d, 0.0d]
    geopack_t01, parmod_t01s, -4.0d, 2.0d, 1.0d, t01s_bx, t01s_by, t01s_bz, /storm

    ; 16. T96 magnetopause – point inside (3, 0, 0) RE (dayside, well inside ~10 RE nose)
    print, '  16. geopack_t96_mgnp_08 (inside)'
    geopack_t96_mgnp_08, 5.0d, -400.0d, 3.0d, 0.0d, 0.0d, $
        t96mgnp_in_xm, t96mgnp_in_ym, t96mgnp_in_zm, t96mgnp_in_dist, t96mgnp_in_id

    ; 17. T96 magnetopause – point outside (25, 0, 0) RE (beyond the ~10 RE subsolar nose)
    print, '  17. geopack_t96_mgnp_08 (outside)'
    geopack_t96_mgnp_08, 5.0d, -400.0d, 25.0d, 0.0d, 0.0d, $
        t96mgnp_out_xm, t96mgnp_out_ym, t96mgnp_out_zm, t96mgnp_out_dist, t96mgnp_out_id

    ; 18. Shue et al. (1997) magnetopause – inside (3, 0, 0) RE
    print, '  18. geopack_shuetal_mgnp_08 (inside)'
    geopack_shuetal_mgnp_08, 5.0d, -400.0d, -5.0d, 3.0d, 0.0d, 0.0d, $
        shue_in_xm, shue_in_ym, shue_in_zm, shue_in_dist, shue_in_id

    ; 19. Shue et al. (1997) magnetopause – outside (25, 0, 0) RE
    print, '  19. geopack_shuetal_mgnp_08 (outside)'
    geopack_shuetal_mgnp_08, 5.0d, -400.0d, -5.0d, 25.0d, 0.0d, 0.0d, $
        shue_out_xm, shue_out_ym, shue_out_zm, shue_out_dist, shue_out_id

    ; 20. Field-line trace from (4, 0, 0) RE, T89 + IGRF
    ;    dir=-1.0d in IDL geopack_trace_08 convention = trace parallel to B
    ;              which matches Python geopack.trace(..., dir=+1)
    ;    par=3     = iopt for T89 (Kp index)
    print, '  20. geopack_trace_08'
    geopack_trace_08, -4.0d, 0.0d, 0.0d, -1.0d, 3, trace_xf, trace_yf, trace_zf, $
        fline=trace_fline, /igrf, /t89, r0=1.0, rlim=60.0, dsmax=0.5, err=0.0001
    print, sqrt(total(trace_fline^2.0, 2))
    
    ; ---- Save all reference values ----
    print, 'Saving test_geopack.sav...'
    save, filename='/Users/xnchu/Library/CloudStorage/Dropbox/projects/IRBEM/python/geopack/test_geopack.sav', $
        tilt, $
        igrf_geo_br, igrf_geo_bt, igrf_geo_bp, $
        igrf_gsw_bx, igrf_gsw_by, igrf_gsw_bz, $
        dip_bx, dip_by, dip_bz, $
        sphcar_x, sphcar_y, sphcar_z, $
        sphcar_r, sphcar_theta, sphcar_phi, $
        bspcar_bx, bspcar_by, bspcar_bz, $
        bcarsp_br, bcarsp_bt, bcarsp_bp, $
        cc_geo2gsm_d1, cc_geo2gsm_d2, cc_geo2gsm_d3, $
        cc_gsm2geo_d1, cc_gsm2geo_d2, cc_gsm2geo_d3, $
        cc_geo2gse_d1, cc_geo2gse_d2, cc_geo2gse_d3, $
        t89_bx, t89_by, t89_bz, $
        t96_bx, t96_by, t96_bz, $
        t01_bx, t01_by, t01_bz, $
        ts04_bx, ts04_by, ts04_bz, $
        t01s_bx, t01s_by, t01s_bz, $
        t96mgnp_in_xm, t96mgnp_in_ym, t96mgnp_in_zm, t96mgnp_in_dist, t96mgnp_in_id, $
        t96mgnp_out_xm, t96mgnp_out_ym, t96mgnp_out_zm, t96mgnp_out_dist, t96mgnp_out_id, $
        shue_in_xm, shue_in_ym, shue_in_zm, shue_in_dist, shue_in_id, $
        shue_out_xm, shue_out_ym, shue_out_zm, shue_out_dist, shue_out_id, $
        trace_xf, trace_yf, trace_zf, trace_fline

    print, 'Done. Reference values written to test_geopack.sav'
end
