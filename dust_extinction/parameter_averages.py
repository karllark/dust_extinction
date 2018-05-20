from __future__ import (absolute_import, print_function, division)

import numpy as np

import astropy.units as u

from .baseclasses import (BaseExtRvModel, BaseExtRvAfAModel)
from .helpers import (_test_valid_x_range, _curve_F99_method)
from .averages import G03_SMCBar

__all__ = ['CCM89', 'O94', 'F99', 'F04', 'G16']

x_range_CCM89 = [0.3, 10.0]
x_range_O94 = x_range_CCM89
x_range_F99 = [0.3, 10.0]
x_range_F04 = [0.3, 10.0]
x_range_G16 = [0.3, 10.0]


class CCM89(BaseExtRvModel):
    """
    CCM89 extinction model calculation

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    CCM89 Milky Way R(V) dependent extinction model

    From Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245)

    Example showing CCM89 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import CCM89

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.5,10.0,0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = CCM89(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """
    Rv_range = [2.0, 6.0]
    x_range = x_range_CCM89

    @staticmethod
    def evaluate(in_x, Rv):
        """
        CCM89 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # convert to wavenumbers (1/micron) if x input in units
        # otherwise, assume x in appropriate wavenumber units
        with u.add_enabled_equivalencies(u.spectral()):
            x_quant = u.Quantity(in_x, 1.0/u.micron, dtype=np.float64)

        # strip the quantity to avoid needing to add units to all the
        #    polynomical coefficients
        x = x_quant.value

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_CCM89, 'CCM89')

        # setup the a & b coefficient vectors
        n_x = len(x)
        a = np.zeros(n_x)
        b = np.zeros(n_x)

        # define the ranges
        ir_indxs = np.where(np.logical_and(0.3 <= x, x < 1.1))
        opt_indxs = np.where(np.logical_and(1.1 <= x, x < 3.3))
        nuv_indxs = np.where(np.logical_and(3.3 <= x, x <= 8.0))
        fnuv_indxs = np.where(np.logical_and(5.9 <= x, x <= 8))
        fuv_indxs = np.where(np.logical_and(8 < x, x <= 10))

        # Infrared
        y = x[ir_indxs]**1.61
        a[ir_indxs] = .574*y
        b[ir_indxs] = -0.527*y

        # NIR/optical
        y = x[opt_indxs] - 1.82
        a[opt_indxs] = np.polyval((.32999, -.7753, .01979, .72085, -.02427,
                                   -.50447, .17699, 1), y)
        b[opt_indxs] = np.polyval((-2.09002, 5.3026, -.62251, -5.38434,
                                   1.07233, 2.28305, 1.41338, 0), y)

        # NUV
        a[nuv_indxs] = (1.752-.316*x[nuv_indxs]
                        - 0.104/((x[nuv_indxs] - 4.67)**2 + .341))
        b[nuv_indxs] = (-3.09
                        + 1.825*x[nuv_indxs]
                        + 1.206/((x[nuv_indxs] - 4.62)**2 + .263))

        # far-NUV
        y = x[fnuv_indxs] - 5.9
        a[fnuv_indxs] += -.04473*(y**2) - .009779*(y**3)
        b[fnuv_indxs] += .2130*(y**2) + .1207*(y**3)

        # FUV
        y = x[fuv_indxs] - 8.0
        a[fuv_indxs] = np.polyval((-.070, .137, -.628, -1.073), y)
        b[fuv_indxs] = np.polyval((.374, -.42, 4.257, 13.67), y)

        # return A(x)/A(V)
        return a + b/Rv


class O94(BaseExtRvModel):
    """
    O94 extinction model calculation

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    O94 Milky Way R(V) dependent extinction model

    From O'Donnell (1994, ApJ, 422, 158)
      Updates/improves the optical portion of the CCM89 model

    Example showing O94 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import O94

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.5,10.0,0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = O94(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """
    Rv_range = [2.0, 6.0]
    x_range = x_range_O94

    @staticmethod
    def evaluate(in_x, Rv):
        """
        O94 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # convert to wavenumbers (1/micron) if x input in units
        # otherwise, assume x in appropriate wavenumber units
        with u.add_enabled_equivalencies(u.spectral()):
            x_quant = u.Quantity(in_x, 1.0/u.micron, dtype=np.float64)

        # strip the quantity to avoid needing to add units to all the
        #    polynomical coefficients
        x = x_quant.value

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_O94, 'O94')

        # setup the a & b coefficient vectors
        n_x = len(x)
        a = np.zeros(n_x)
        b = np.zeros(n_x)

        # define the ranges
        ir_indxs = np.where(np.logical_and(0.3 <= x, x < 1.1))
        opt_indxs = np.where(np.logical_and(1.1 <= x, x < 3.3))
        nuv_indxs = np.where(np.logical_and(3.3 <= x, x <= 8.0))
        fnuv_indxs = np.where(np.logical_and(5.9 <= x, x <= 8))
        fuv_indxs = np.where(np.logical_and(8 < x, x <= 10))

        # Infrared
        y = x[ir_indxs]**1.61
        a[ir_indxs] = .574*y
        b[ir_indxs] = -0.527*y

        # NIR/optical
        y = x[opt_indxs] - 1.82
        a[opt_indxs] = np.polyval((-0.505, 1.647, -0.827, -1.718,
                                   1.137, 0.701, -0.609, 0.104, 1), y)
        b[opt_indxs] = np.polyval((3.347, -10.805, 5.491, 11.102,
                                   -7.985, -3.989, 2.908, 1.952, 0), y)

        # NUV
        a[nuv_indxs] = (1.752-.316*x[nuv_indxs]
                        - 0.104/((x[nuv_indxs] - 4.67)**2 + .341))
        b[nuv_indxs] = (-3.09
                        + 1.825*x[nuv_indxs]
                        + 1.206/((x[nuv_indxs] - 4.62)**2 + .263))

        # far-NUV
        y = x[fnuv_indxs] - 5.9
        a[fnuv_indxs] += -.04473*(y**2) - .009779*(y**3)
        b[fnuv_indxs] += .2130*(y**2) + .1207*(y**3)

        # FUV
        y = x[fuv_indxs] - 8.0
        a[fuv_indxs] = np.polyval((-.070, .137, -.628, -1.073), y)
        b[fuv_indxs] = np.polyval((.374, -.42, 4.257, 13.67), y)

        # return A(x)/A(V)
        return a + b/Rv


class F99(BaseExtRvModel):
    """
    F99 extinction model calculation

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    F99 Milky Way R(V) dependent extinction model

    From Fitzpatrick (1999, PASP, 111, 63)

    Example showing F99 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import F99

        fig, ax = plt.subplots()

        # temp model to get the correct x range
        text_model = F99()

        # generate the curves and plot them
        x = np.arange(text_model.x_range[0],
                      text_model.x_range[1],0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = F99(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """
    Rv_range = [2.0, 6.0]
    x_range = x_range_F99

    def evaluate(self, in_x, Rv):
        """
        F99 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # ensure Rv is a single element, not numpy array
        Rv = Rv[0]

        # constant terms
        C3 = 3.23
        C4 = 0.41
        xo = 4.596
        gamma = 0.99

        # terms depending on Rv
        C2 = -0.824 + 4.717/Rv
        # original F99 C1-C2 correlation
        C1 = 2.030 - 3.007*C2

        # spline points
        optnir_axav_x = 10000./np.array([26500.0, 12200.0, 6000.0,
                                         5470.0, 4670.0, 4110.0])

        # determine optical/IR values at spline points
        #    Final optical spline point has a leading "-1.208" in Table 4
        #    of F99, but that does not reproduce Table 3.
        #    Additional indication that this is not correct is from
        #    fm_unred.pro
        #    which is based on FMRCURVE.pro distributed by Fitzpatrick.
        #    --> confirmation needed?
        #
        #    Also, fm_unred.pro has different coeff and # of terms,
        #    but later work does not include these terms
        #    --> check with Fitzpatrick?
        opt_axebv_y = np.array([-0.426 + 1.0044*Rv,
                                -0.050 + 1.0016*Rv,
                                0.701 + 1.0016*Rv,
                                1.208 + 1.0032*Rv - 0.00033*(Rv**2)])
        nir_axebv_y = np.array([0.265, 0.829])*Rv/3.1
        optnir_axebv_y = np.concatenate([nir_axebv_y, opt_axebv_y])

        # return A(x)/A(V)
        return _curve_F99_method(in_x, Rv, C1, C2, C3, C4, xo, gamma,
                                 optnir_axav_x, optnir_axebv_y/Rv,
                                 self.x_range, 'F99')


class F04(BaseExtRvModel):
    """
    F99 extinction model calculation

    Updated with the NIR Rv dependence in
       Fitzpatrick (2004, ASP Conf. Ser. 309, Astrophysics of Dust, 33)

    See also Fitzpatrick & Massa (2007, ApJ, 663, 320)

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    F99 Milky Way R(V) dependent extinction model

    From Fitzpatrick (1999, PASP, 111, 63)

    Updated with the NIR Rv dependence in
       Fitzpatrick (2004, ASP Conf. Ser. 309, Astrophysics of Dust, 33)

    See also Fitzpatrick & Massa (2007, ApJ, 663, 320)

    Example showing F04 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import F04

        fig, ax = plt.subplots()

        # temp model to get the correct x range
        text_model = F04()

        # generate the curves and plot them
        x = np.arange(text_model.x_range[0],
                      text_model.x_range[1],0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = F04(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """
    Rv_range = [2.0, 6.0]
    x_range = x_range_F04

    def evaluate(self, in_x, Rv):
        """
        F04 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # ensure Rv is a single element, not numpy array
        Rv = Rv[0]

        # constant terms
        C3 = 2.991
        C4 = 0.319
        xo = 4.592
        gamma = 0.922

        # original F99 Rv dependence
        C2 = -0.824 + 4.717/Rv
        # updated F04 C1-C2 correlation
        C1 = 2.18 - 2.91*C2

        # spline points
        opt_axav_x = 10000./np.array([6000.0, 5470.0,
                                      4670.0, 4110.0])
        # **Use NIR spline x values in FM07, clipped to K band for now
        nir_axav_x = np.array([0.50, 0.75, 1.0])
        optnir_axav_x = np.concatenate([nir_axav_x, opt_axav_x])

        # **Keep optical spline points from F99:
        #    Final optical spline point has a leading "-1.208" in Table 4
        #    of F99, but that does not reproduce Table 3.
        #    Additional indication that this is not correct is from
        #    fm_unred.pro
        #    which is based on FMRCURVE.pro distributed by Fitzpatrick.
        #    --> confirmation needed?
        opt_axebv_y = np.array([-0.426 + 1.0044*Rv,
                                -0.050 + 1.0016*Rv,
                                0.701 + 1.0016*Rv,
                                1.208 + 1.0032*Rv - 0.00033*(Rv**2)])
        # updated NIR curve from F04, note R dependendence
        nir_axebv_y = (0.63*Rv - 0.84)*nir_axav_x**1.84

        optnir_axebv_y = np.concatenate([nir_axebv_y, opt_axebv_y])

        # return A(x)/A(V)
        return _curve_F99_method(in_x, Rv, C1, C2, C3, C4, xo, gamma,
                                 optnir_axav_x, optnir_axebv_y/Rv,
                                 self.x_range, 'F04')


class G16(BaseExtRvAfAModel):
    """
    G16 extinction model calculation

    Mixture model between the F99 R(V) dependent model (component A)
    and the G03_SMCBar model (component B)

    Parameters
    ----------
    RvA: float
         R_A(V) = A(V)/E(B-V) = total-to-selective extinction
         R(V) of the A component

    fA: float
        f_A is the mixture coefficent between the R(V)

    Raises
    ------
    InputParameterError
       Input RvA values outside of defined range
       Input fA values outside of defined range

    Notes
    -----
    G16 R_A(V) and f_A dependent model

    From Gordon et al. (2016, ApJ, 826, 104)

    Example showing G16 curves for a range of R_A(V) values
    and f_A values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import G16

        fig, ax = plt.subplots()

        # temp model to get the correct x range
        text_model = G16()

        # generate the curves and plot them
        x = np.arange(text_model.x_range[0],
                      text_model.x_range[1],0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = G16(RvA=cur_Rv, fA=1.0)
           ax.plot(x,ext_model(x),label=r'$R_A(V) = ' + str(cur_Rv) + '$')

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best', title=r'$f_A = 1.0$')
        plt.show()

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import G16

        fig, ax = plt.subplots()

        # temp model to get the correct x range
        text_model = G16()

        # generate the curves and plot them
        x = np.arange(text_model.x_range[0],
                      text_model.x_range[1],0.1)/u.micron

        fAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for cur_fA in fAs:
           ext_model = G16(RvA=3.1, fA=cur_fA)
           ax.plot(x,ext_model(x),label=r'$f_A = ' + str(cur_fA) + '$')

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best', title=r'$R_A(V) = 3.1$')
        plt.show()
    """

    RvA_range = [2.0, 6.0]
    fA_range = [0.0, 1.0]
    x_range = x_range_G16

    @staticmethod
    def evaluate(in_x, RvA, fA):
        """
        G16 function

        Parameters
        ----------
        in_x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # convert to wavenumbers (1/micron) if x input in units
        # otherwise, assume x in appropriate wavenumber units
        with u.add_enabled_equivalencies(u.spectral()):
            x_quant = u.Quantity(in_x, 1.0/u.micron, dtype=np.float64)

        # strip the quantity to avoid needing to add units to all the
        #    polynomical coefficients
        x = x_quant.value

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_G16, 'G16')

        # ensure Rv is a single element, not numpy array
        RvA = RvA[0]

        # get the A component extinction model
        extA_model = F99(Rv=RvA)
        alav_A = extA_model(x)

        # get the B component extinction model
        extB_model = G03_SMCBar()
        alav_B = extB_model(x)

        # create the mixture model
        alav = fA*alav_A + (1.0 - fA)*alav_B

        # return A(x)/A(V)
        return alav
