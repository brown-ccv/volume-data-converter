import numpy as np
import typer


def stretching_parameters(kgrid, Np, N):
    """
        Auxiliar function to initialize stretching data
    """
    ds = 1.0 / N
    if kgrid == 1:
        Nlev = Np
        lev = np.arange(N)
        s = np.multiply(lev - N, ds)
    else:
        Nlev = N
        lev = np.arange(1, N + 1) - 0.5
        s = np.multiply(lev - N, ds)
    return Nlev, s


def stretching(Vstretching, theta_s, theta_b, hc, N, kgrid, report=False):
    """
     STRETCHING:  Compute ROMS vertical coordinate stretching function
    
    
     [s,C]=stretching(*data)
       Data is a list of arguments that represents the following:

       data[0]: Vstretching
       data[1]: theta_s
       data[2]: theta_b
       data[3]: hc
       data[4]: N
       data[5]: kgrid
       data[6]: report = False by default
    
       i.e:
       stretching(Vstretching, theta_s, theta_b, hc, N, kgrid, False)

       Given vertical terrain-following vertical stretching parameters, this
       routine computes the vertical stretching function used in ROMS vertical
       coordinate transformation. Check the following link for details:
    
       https://www.myroms.org/wiki/index.php/Vertical_S-coordinate
    
       On Input:
    
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
                        kgrid = 0,        function at vertical RHO-points
                        kgrid = 1,        function at vertical W-points
        report        Flag to report detailed information (OPTIONAL):
                        report = false,   do not report
                        report = true,    report information
    
       On Output:
    
       s             S-coordinate independent variable, [-1 <= s <= 0] at
                        vertical RHO- or W-points (vector)
       C             Nondimensional, monotonic, vertical stretching function,
                        C(s), 1D array, [-1 <= C(s) <= 0]
    

    svn $Id: stretching.m 711 2014-01-23 20:36:13Z arango $
    =========================================================================%
      Copyright (c) 2002-2014 The ROMS/TOMS Group                            %
        Licensed under a MIT/X style license                                 %
        See License_ROMS.txt                           Hernan G. Arango      %
    =========================================================================%
    """

    s = np.array([])
    C = np.array([])
    # --------------------------------------------------------------------------
    #   Set several parameters.
    # --------------------------------------------------------------------------

    if Vstretching < 1 or Vstretching > 4:
        typer.echo(" ")
        typer.echo(
            "*** Error:  STRETCHING - Illegal parameter Vstretching = "
            + str(Vstretching)
        )
        typer.echo(" ")
        return

    Np = N + 1
    Nlev = 0
    # --------------------------------------------------------------------------
    #  Compute ROMS S-coordinates vertical stretching function
    # --------------------------------------------------------------------------

    if Vstretching == 1:
        Nlev, s = stretching_parameters(kgrid, Np, N)
        if theta_s > 0:
            Ptheta = np.divide(np.sinh(theta_s * s), np.sinh(theta_s))
            Rtheta = np.divide(
                np.tanh(theta_s * (s + 0.5)), (2.0 * np.tanh(0.5 * theta_s)) - 0.5
            )
            C = (1.0 - theta_b) * Ptheta + theta_b * Rtheta
        else:
            C = s
    #  A. Shchepetkin (UCLA-ROMS, 2005) vertical stretching function.
    elif Vstretching == 2:
        alfa = 1.0
        beta = 1.0
        Nlev, s = stretching_parameters(kgrid, Np, N)
        if theta_s > 0:
            Csur = np.divide(1.0 - np.cosh(theta_s * s), (np.cosh(theta_s) - 1.0))
            if theta_b > 0:
                Cbot = -1.0 + np.divide(np.sinh(theta_b * (s + 1.0)), np.sinh(theta_b))
                weigth = np.multiply(
                    np.multiply(np.power(s + 1.0, alfa), 1.0 + (alfa / beta)),
                    (np.power(1.0 - (s + 1.0), beta)),
                )
                C = weigth * Csur + (1.0 - weigth) * Cbot
            else:
                C = Csur
        else:
            C = s

    #   R. Geyer BBL vertical stretching function.
    elif Vstretching == 3:
        Nlev, s = stretching_parameters(kgrid, Np, N)
        if theta_s > 0:
            exp_s = theta_s  # surface stretching exponent
            exp_b = theta_b  # bottom  stretching exponent
            alpha = 3  # scale factor for all hyperbolic functions
            Cbot = (
                np.divide(
                    np.log(np.cosh(np.power(alpha * (s + 1), exp_b))),
                    np.log(np.cosh(alpha)),
                )
                - 1
            )
            Csur = np.divide(
                -np.log(
                    np.cosh(np.power(alpha * np.abs(s), exp_s)), np.log(np.cosh(alpha))
                )
            )
            weight = (1 - np.tanh(alpha * (s + 0.5))) / 2
            C = weight * Cbot + (1 - weight) * Csur
        else:
            C = s

    #  A. Shchepetkin (UCLA-ROMS, 2010) double vertical stretching function
    #  with bottom refinement

    elif Vstretching == 4:
        Nlev, s = stretching_parameters(kgrid, Np, N)
        if theta_s > 0:
            Csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
        else:
            Csur = np.power(-s, 2)

        if theta_b > 0:
            Cbot = (np.exp(theta_b * Csur) - 1.0) / (1.0 - np.exp(-theta_b))
            C = Cbot
        else:
            C = Csur

    #  Report S-coordinate parameters.

    if report:
        typer.echo(" ")
        if Vstretching == 1:
            typer.echo(
                "Vstretching = " + str(Vstretching) + "   Song and Haidvogel (1994)"
            )
        elif Vstretching == 2:
            typer.echo("Vstretching = " + str(Vstretching) + "   Shchepetkin (2005)")
        elif Vstretching == 3:
            typer.echo("Vstretching = " + str(Vstretching) + "   Geyer (2009), BBL")
        elif Vstretching == 4:
            typer.echo("Vstretching = " + str(Vstretching) + "   Shchepetkin (2010)")

        if kgrid == 1:
            typer.echo("   kgrid    = " + str(kgrid) + "   at vertical W-points")
        else:
            typer.echo("   kgrid    = " + str(kgrid) + "   at vertical RHO-points")

        typer.echo("   theta_s  = " + str(theta_s))
        typer.echo("   theta_b  = " + str(theta_b))
        typer.echo("   hc       = " + str(hc))
        typer.echo(" ")
        typer.echo(" S-coordinate curves: k, s(k), C(k)")
        typer.echo(" ")

        if kgrid == 1:
            # for k = Nlev:-1:1:
            for k in range(Nlev, 0, -1):
                k_str = "{:.3g}".format(k - 1)
                s_k = "{:20.12e}".format(s[k])
                c_k = "{:20.12e}".format(C[k])
                typer.echo("    " + k_str + "   " + s_k + "   " + c_k)

        else:
            for k in range(Nlev - 1, 0, -1):
                k_str = "{:.3g}".format(k)
                s_k = "{:20.12e}".format(s[k])
                c_k = "{:20.12e}".format(C[k])
                typer.echo("    " + k_str + "   " + s_k + "   " + c_k)

    typer.echo(" ")

    return [s, C]
