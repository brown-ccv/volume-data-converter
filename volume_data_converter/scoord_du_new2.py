from volume_data_converter import stretching as stch
import numpy as np
import typer


def scoord_du_new2(h, zeta, Vtransform, Vstretching, theta_s, theta_b, hc, N, kgrid):
    """

     SCOORD:  Compute and plot ROMS vertical stretched coordinates
    
     [z,s,C]=scoord_du_new2(h, zeta, Vtransform, Vstretching, theta_s, theta_b, hc, N, kgrid)
    
     Given a batymetry (h), surface elevation (zeta) and terrain-following stretching parameters,
     this function computes the depths of RHO- or W-points for all grid points. Check the
     following link for details:
    
        https://www.myroms.org/wiki/index.php/Vertical_S-coordinate
    
     On Input:
    
        h             Bottom depth 2D array, h(1:Lp,1:Mp), m, positive
        zeta          surface elevation
        Vtransform    Vertical transformation equation:
                       Vtransform = 1,   original transformation
    
                          z(x,y,s,t)=Zo(x,y,s)+zeta(x,y,t)*[1+Zo(x,y,s)/h(x,y)]
    
                          Zo(x,y,s)=hc*s+[h(x,y)-hc]*C(s)
    
                       Vtransform = 2,   new transformation
    
                          z(x,y,s,t)=zeta(x,y,t)+[zeta(x,y,t)+h(x,y)]*Zo(x,y,s)
    
                           Zo(x,y,s)=[hc*s(k)+h(x,y)*C(k)]/[hc+h(x,y)]
        Vstretching   Vertical stretching function:
                        Vstretching = 1,  original (Song and Haidvogel, 1994)
                        Vstretching = 2,  A. Shchepetkin (UCLA-ROMS, 2005)
                        Vstretching = 3,  R. Geyer BBL refinement
                        Vstretching = 4,  A. Shchepetkin (UCLA-ROMS, 2010)
        theta_s       S-coordinate surface control parameter (scalar)
        theta_b       S-coordinate bottom control parameter (scalar)
        hc            Width (m) of surface or bottom boundary layer in which
                        higher vertical resolution is required during
                       stretching (scalar)
        N             Number of vertical levels (scalar)
        kgrid         Depth grid type logical switch:
                        kgrid = 0,        depths of RHO-points
                        kgrid = 1,        depths of W-points
    
    On Output:
    
        z             Depths (m) of RHO- or W-points (matrix)
        s             S-coordinate independent variable, [-1 <= s <= 0] at
                       vertical RHO- or W-points (vector)
        C             Nondimensional, monotonic, vertical stretching function,
                        C(s), 1D array, [-1 <= C(s) <= 0]z
    

    svn $Id: scoord.m 754 2015-01-07 23:23:40Z arango $
    ===========================================================================%
      Copyright (c) 2002-2015 The ROMS/TOMS Group                              %
        Licensed under a MIT/X style license                                   %
        See License_ROMS.txt                           Hernan G. Arango        %
    ===========================================================================%
     10/4/2018: DU modified scoord.m in ROMS_matlab_11112015 to remove plotting, verbose output, and
                to allow input of zeta (original version assumed zero zeta).
                this version does the computation for all gridpoints rather than those along
             a row or column as the original function did.
    
    """

    # ----------------------------------------------------------------------------
    #   Set several parameters.
    # ----------------------------------------------------------------------------

    if hc > np.min(np.min(h)) and Vtransform == 1:
        typer.echo(" ")
        typer.echo(
            "*** Error:  SCOORD - critical depth exceeds minimum bathymetry value."
        )
        typer.echo("Vtranform = " + str(Vtransform))
        typer.echo("hc        = " + str(hc))
        typer.echo("hmax      = " + str(min(min(h))))
        typer.echo(" ")
        return

    if Vtransform < 1 or Vtransform > 2:
        typer.echo(" ")
        typer.echo(
            "*** Error:  SCOORD - Illegal parameter Vtransform = " + str(Vtransform)
        )
        return

    if Vstretching < 1 or Vstretching > 4:
        typer.echo(" ")
        typer.echo(
            "*** Error:  SCOORD - Illegal parameter Vstretching = " + str(Vstretching)
        )
        return

    hShape = h.shape
    Lp = hShape[0]
    Mp = hShape[1]
    hmin = np.min(np.min(h))
    hmax = np.max(np.max(h))
    havg = 0.5 * (hmax + hmin)
    # ----------------------------------------------------------------------------
    #  Compute vertical stretching function, C(k):
    # ----------------------------------------------------------------------------

    [s, C] = stch.stretching(Vstretching, theta_s, theta_b, hc, N, kgrid, 0)

    if kgrid == 1:
        Nlev = N + 1
    else:
        Nlev = N

    zhc = np.zeros(Nlev)
    z1 = np.zeros(Nlev)
    z2 = np.zeros(Nlev)
    z3 = np.zeros(Nlev)

    if Vtransform == 1:

        for k in reversed(range(Nlev)):
            zhc = hc * s[k]
            zhc2 = zhc[k]
            z1_temp = zhc2[k] + (hmin - hc) * C[k]
            z2_temp = zhc2[k] + (havg - hc) * C[k]
            z3_temp = zhc2[k] + (hmax - hc) * C[k]
            z1 = z1_temp[k]
            z2 = z2_temp[k]
            z3 = z3_temp[k]

    elif Vtransform == 2:

        for k in reversed(range(Nlev)):
            if hc > hmax:
                zhc[k] = hmax * (hc * s[k] + hmax * C[k]) / (hc + hmax)
            else:
                zhc[k] = 0.5 * np.minimum(hc, hmax) * (s[k] + C[k])

            z1[k] = hmin * (hc * s[k] + hmin * C[k]) / (hc + hmin)
            z2[k] = havg * (hc * s[k] + havg * C[k]) / (hc + havg)
            z3[k] = hmax * (hc * s[k] + hmax * C[k]) / (hc + hmax)

    # ============================================================================
    #  Compute depths at all gridpoints
    # ============================================================================
    z = np.ma.zeros((Nlev, Lp, Mp))
    if Vtransform == 1:

        for k in range(Nlev):
            z0 = hc * (s[k] - C[k]) + h * C(k)
            z[k, :, :] = z0 + zeta * (1.0 + z0 / h)
    elif Vtransform == 2:

        for k in range(Nlev):
            t = np.around(s[k], 4)
            u = np.ma.round(C[k], 4)
            z0 = np.ma.round((hc * t + u * h) / (h + hc), 4)
            z[k, :, :] = np.ma.round(zeta + (zeta + h) * z0, decimals=4)

    return [z, s, C]
