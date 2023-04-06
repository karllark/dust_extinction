from __future__ import absolute_import, print_function, division

import pkg_resources

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.table import Table
from astropy.modeling.models import Drude1D, Polynomial1D, PowerLaw1D

from .baseclasses import BaseExtRvModel, BaseExtRvAfAModel
from .helpers import _get_x_in_wavenumbers, _test_valid_x_range, _smoothstep
from .averages import G03_SMCBar
from .shapes import _curve_F99_method, _modified_drude, FM90

# fmt: off
__all__ = ["CCM89", "O94", "F99", "F04", "VCG04", "GCC09", "M14", "G16", "F19",
           "D22", "G23"]
# fmt: on

x_range_CCM89 = [0.3, 10.0]
x_range_O94 = x_range_CCM89
x_range_F99 = [0.3, 10.0]
x_range_F04 = [0.3, 10.0]
x_range_VCG04 = [3.3, 8.0]
x_range_GCC09 = [3.3, 11.0]
x_range_M14 = [0.3, 3.3]
x_range_G16 = [0.3, 10.0]
x_range_G23 = [1.0 / 32.0, 1.0 / 0.0912]


class CCM89(BaseExtRvModel):
    r"""
    Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model

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

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_CCM89, "CCM89")

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
        y = x[ir_indxs] ** 1.61
        a[ir_indxs] = 0.574 * y
        b[ir_indxs] = -0.527 * y

        # NIR/optical
        y = x[opt_indxs] - 1.82
        a[opt_indxs] = np.polyval(
            (0.32999, -0.7753, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1), y
        )
        b[opt_indxs] = np.polyval(
            (-2.09002, 5.3026, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0), y
        )

        # NUV
        a[nuv_indxs] = (
            1.752 - 0.316 * x[nuv_indxs] - 0.104 / ((x[nuv_indxs] - 4.67) ** 2 + 0.341)
        )
        b[nuv_indxs] = (
            -3.09 + 1.825 * x[nuv_indxs] + 1.206 / ((x[nuv_indxs] - 4.62) ** 2 + 0.263)
        )

        # far-NUV
        y = x[fnuv_indxs] - 5.9
        a[fnuv_indxs] += -0.04473 * (y**2) - 0.009779 * (y**3)
        b[fnuv_indxs] += 0.2130 * (y**2) + 0.1207 * (y**3)

        # FUV
        y = x[fuv_indxs] - 8.0
        a[fuv_indxs] = np.polyval((-0.070, 0.137, -0.628, -1.073), y)
        b[fuv_indxs] = np.polyval((0.374, -0.42, 4.257, 13.67), y)

        # return A(x)/A(V)
        return a + b / Rv


class O94(BaseExtRvModel):
    r"""
    O'Donnell (1994) Milky Way R(V) dependent model

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

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_O94, "O94")

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
        y = x[ir_indxs] ** 1.61
        a[ir_indxs] = 0.574 * y
        b[ir_indxs] = -0.527 * y

        # NIR/optical
        y = x[opt_indxs] - 1.82
        a[opt_indxs] = np.polyval(
            (-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1), y
        )
        b[opt_indxs] = np.polyval(
            (3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0), y
        )

        # NUV
        a[nuv_indxs] = (
            1.752 - 0.316 * x[nuv_indxs] - 0.104 / ((x[nuv_indxs] - 4.67) ** 2 + 0.341)
        )
        b[nuv_indxs] = (
            -3.09 + 1.825 * x[nuv_indxs] + 1.206 / ((x[nuv_indxs] - 4.62) ** 2 + 0.263)
        )

        # far-NUV
        y = x[fnuv_indxs] - 5.9
        a[fnuv_indxs] += -0.04473 * (y**2) - 0.009779 * (y**3)
        b[fnuv_indxs] += 0.2130 * (y**2) + 0.1207 * (y**3)

        # FUV
        y = x[fuv_indxs] - 8.0
        a[fuv_indxs] = np.polyval((-0.070, 0.137, -0.628, -1.073), y)
        b[fuv_indxs] = np.polyval((0.374, -0.42, 4.257, 13.67), y)

        # return A(x)/A(V)
        return a + b / Rv


class F99(BaseExtRvModel):
    r"""
    Fitzpatrick (1999) Milky Way R(V) dependent model

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

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

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
        # just in case someone calls evaluate explicitly
        Rv = np.atleast_1d(Rv)

        # ensure Rv is a single element, not numpy array
        Rv = Rv[0]

        # constant terms
        C3 = 3.23
        C4 = 0.41
        xo = 4.596
        gamma = 0.99

        # terms depending on Rv
        C2 = -0.824 + 4.717 / Rv
        # original F99 C1-C2 correlation
        C1 = 2.030 - 3.007 * C2

        # spline points
        optnir_axav_x = 10000.0 / np.array(
            [26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0]
        )

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
        opt_axebv_y = np.array(
            [
                -0.426 + 1.0044 * Rv,
                -0.050 + 1.0016 * Rv,
                0.701 + 1.0016 * Rv,
                1.208 + 1.0032 * Rv - 0.00033 * (Rv**2),
            ]
        )
        nir_axebv_y = np.array([0.265, 0.829]) * Rv / 3.1
        optnir_axebv_y = np.concatenate([nir_axebv_y, opt_axebv_y])

        # return A(x)/A(V)
        return _curve_F99_method(
            in_x,
            Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axebv_y / Rv,
            self.x_range,
            self.__class__.__name__,
        )


class F04(BaseExtRvModel):
    r"""
    Fitzpatrick (2004) Milky Way R(V) dependent model

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
    From Fitzpatrick (2004, ASP Conf. Ser. 309, Astrophysics of Dust, 33)
        Equivalent to the F99 model with an updated NIR Rv dependence

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

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

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
        # just in case someone calls evaluate explicitly
        Rv = np.atleast_1d(Rv)

        # ensure Rv is a single element, not numpy array
        Rv = Rv[0]

        # constant terms
        C3 = 2.991
        C4 = 0.319
        xo = 4.592
        gamma = 0.922

        # original F99 Rv dependence
        C2 = -0.824 + 4.717 / Rv
        # updated F04 C1-C2 correlation
        C1 = 2.18 - 2.91 * C2

        # spline points
        opt_axav_x = 10000.0 / np.array([6000.0, 5470.0, 4670.0, 4110.0])
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
        opt_axebv_y = np.array(
            [
                -0.426 + 1.0044 * Rv,
                -0.050 + 1.0016 * Rv,
                0.701 + 1.0016 * Rv,
                1.208 + 1.0032 * Rv - 0.00033 * (Rv**2),
            ]
        )
        # updated NIR curve from F04, note R dependence
        nir_axebv_y = (0.63 * Rv - 0.84) * nir_axav_x**1.84

        optnir_axebv_y = np.concatenate([nir_axebv_y, opt_axebv_y])

        # return A(x)/A(V)
        return _curve_F99_method(
            in_x,
            Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axebv_y / Rv,
            self.x_range,
            self.__class__.__name__,
        )


class VCG04(BaseExtRvModel):
    r"""
    Valencic, Clayton, & Gordon (2004) Milky Way R(V) dependent model

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
    From Valencic, Clayton, & Gordon (2004, ApJ, 616, 912)
    Including erratum: 2014, ApJ, 793, 66

    This model applies to the UV spectral region all the way to 912 A.
    This model was not derived for the optical or NIR.

    Example showing V04 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import VCG04

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(3.3,8.0, 0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = VCG04(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    Rv_range = [2.0, 6.0]
    x_range = x_range_VCG04

    @staticmethod
    def evaluate(in_x, Rv):
        """
        VCG04 function

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_VCG04, "VCG04")

        # setup the a & b coefficient vectors
        n_x = len(x)
        a = np.zeros(n_x)
        b = np.zeros(n_x)

        # define the ranges
        nuv_indxs = np.where(np.logical_and(3.3 <= x, x <= 8.0))
        fnuv_indxs = np.where(np.logical_and(5.9 <= x, x <= 8))

        # NUV
        a[nuv_indxs] = (
            1.808 - 0.215 * x[nuv_indxs] - 0.134 / ((x[nuv_indxs] - 4.558) ** 2 + 0.566)
        )
        b[nuv_indxs] = (
            -2.350
            + 1.403 * x[nuv_indxs]
            + 1.103 / ((x[nuv_indxs] - 4.587) ** 2 + 0.263)
        )

        # far-NUV
        y = x[fnuv_indxs] - 5.9
        a[fnuv_indxs] += -0.0077 * (y**2) - 0.0030 * (y**3)
        b[fnuv_indxs] += 0.2060 * (y**2) + 0.0550 * (y**3)

        # return A(x)/A(V)
        return a + b / Rv


class GCC09(BaseExtRvModel):
    r"""
    Grodon, Cartledge, & Clayton (2009) Milky Way R(V) dependent model

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
    From Gordon, Cartledge, & Clayton (2009, ApJ, 705, 1320)
    Including erratum: 2014, ApJ, 781, 128

    This model applies to the UV spectral region all the way to 912 A.
    This model was not derived for the optical or NIR.

    Example showing GCC09 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import GCC09

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(3.3, 11, 0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = GCC09(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    Rv_range = [2.0, 6.0]
    x_range = x_range_GCC09

    @staticmethod
    def evaluate(in_x, Rv):
        """
        GCC09 function

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_GCC09, "GCC09")

        # setup the a & b coefficient vectors
        n_x = len(x)
        a = np.zeros(n_x)
        b = np.zeros(n_x)

        # define the ranges
        nuv_indxs = np.where(np.logical_and(3.3 <= x, x <= 11.0))
        fnuv_indxs = np.where(np.logical_and(5.9 <= x, x <= 11.0))

        # NUV
        a[nuv_indxs] = (
            1.894
            - 0.373 * x[nuv_indxs]
            - 0.0101 / ((x[nuv_indxs] - 4.57) ** 2 + 0.0384)
        )
        b[nuv_indxs] = (
            -3.490 + 2.057 * x[nuv_indxs] + 0.706 / ((x[nuv_indxs] - 4.59) ** 2 + 0.169)
        )

        # far-NUV
        y = x[fnuv_indxs] - 5.9
        a[fnuv_indxs] += -0.110 * (y**2) - 0.0100 * (y**3)
        b[fnuv_indxs] += 0.531 * (y**2) + 0.0544 * (y**3)

        # return A(x)/A(V)
        return a + b / Rv


class M14(BaseExtRvModel):
    r"""
    Maiz Apellaniz et al (2014) Milky Way & LMC R(V) dependent model

    Parameters
    ----------
    R5495: float
        R5495 = A(5485)/E(4405-5495)
        Spectral equivalent to photometric R(V),
        standard value is 3.1

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    M14 R5485-dependent model

    From Maiz Apellaniz et al. (2014, A&A, 564, 63),
    following structure of IDL code provided in paper appendix

    The published UV extinction curve is identical to Clayton, Cardelli,
    and Mathis (1989, CCM). Forcing the optical section to match
    smoothly with CCM introduces a non-physical feature at high
    values of R5495 around 3.9 inverse microns; see section 5 in
    Maiz Apellaniz et al. (2014) for more discussion.  For
    that reason, we provide the M14 model only through 3.3 inverse
    microns, the limit of the optical in CCM.

    R5495 = A(5485)/E(4405-5495)
    Spectral equivalent to photometric R(V),
    standard value is 3.1

    Example showing M14 curves for a range of R5495 values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import M14

        fig, ax = plt.subplots()

        # temp model to get the correct x range
        text_model = M14()

        # generate the curves and plot them
        x = np.arange(text_model.x_range[0],
                      text_model.x_range[1],0.1)/u.micron

        Rvs = ['2.0','3.1','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = M14(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    Rv_range = [2.0, 6.0]
    x_range = x_range_M14

    def evaluate(self, in_x, Rv):
        """
        M14 function

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, self.__class__.__name__)

        # just in case someone calls evaluate explicitly
        Rv = np.atleast_1d(Rv)

        # ensure Rv is a single element, not numpy array
        Rv = Rv[0]

        # Infrared
        ai = 0.574 * x**1.61
        bi = -0.527 * x**1.61

        # Optical
        x1 = np.array([1.0])
        xi1 = x1[0]
        x2 = np.array([1.15, 1.81984, 2.1, 2.27015, 2.7])
        x3 = np.array([3.5, 3.9, 4.0, 4.1, 4.2])
        xi3 = x3[-1]

        a1v = 0.574 * x1**1.61
        a1d = 0.574 * 1.61 * xi1**0.61
        b1v = -0.527 * x1**1.61
        b1d = -0.527 * 1.61 * xi1**0.61

        a2v = (
            1
            + 0.17699 * (x2 - 1.82)
            - 0.50447 * (x2 - 1.82) ** 2
            - 0.02427 * (x2 - 1.82) ** 3
            + 0.72085 * (x2 - 1.82) ** 4
            + 0.01979 * (x2 - 1.82) ** 5
            - 0.77530 * (x2 - 1.82) ** 6
            + 0.32999 * (x2 - 1.82) ** 7
            + np.array([0.0, 0.0, -0.011, 0.0, 0.0])
        )
        b2v = (
            1.41338 * (x2 - 1.82)
            + 2.28305 * (x2 - 1.82) ** 2
            + 1.07233 * (x2 - 1.82) ** 3
            - 5.38434 * (x2 - 1.82) ** 4
            - 0.62251 * (x2 - 1.82) ** 5
            + 5.30260 * (x2 - 1.82) ** 6
            - 2.09002 * (x2 - 1.82) ** 7
            + np.array([0.0, 0.0, +0.091, 0.0, 0.0])
        )

        a3v = (
            1.752
            - 0.316 * x3
            - 0.104 / ((x3 - 4.67) ** 2 + 0.341)
            + np.array([0.442, 0.341, 0.130, 0.020, 0.000])
        )
        a3d = -0.316 + 0.104 * 2.0 * (xi3 - 4.67) / ((xi3 - 4.67) ** 2 + 0.341) ** 2
        b3v = (
            -3.090
            + 1.825 * x3
            + 1.206 / ((x3 - 4.62) ** 2 + 0.263)
            - np.array([1.256, 1.021, 0.416, 0.064, 0.000])
        )
        b3d = 1.825 - 1.206 * 2 * (xi3 - 4.62) / ((xi3 - 4.62) ** 2 + 0.263) ** 2

        xn = np.concatenate((x1, x2, x3))
        anv = np.concatenate((a1v, a2v, a3v))
        bnv = np.concatenate((b1v, b2v, b3v))

        a_spl = interpolate.CubicSpline(xn, anv, bc_type=((1, a1d), (1, a3d)))
        b_spl = interpolate.CubicSpline(xn, bnv, bc_type=((1, b1d), (1, b3d)))

        av = a_spl(x)
        bv = b_spl(x)

        # UV extinction curve in the paper repeats CCM. Forcing the
        # optical section to match smoothly with CCM introduces a
        # non-physical feature at high values of R5495 at x = 3.9
        # inverse microns.  This class does not provide the UV curve,
        # but the code that would calculate it is included below for
        # completeness.

        # Ultraviolet
        # y = x - 5.9
        # fa = np.zeros(x.size)
        #      + (-0.04473*y**2 - 0.009779*y**3)*((x<8)&(x>5.9))
        # fb = np.zeros(x.size) + ( 0.2130*y**2 + 0.1207*y**3)*((x<8)&(x>5.9))

        # au = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) + fa
        # bu = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + fb

        # Far ultraviolet
        # y = x - 8.0
        # af = -1.073 - 0.628*y + 0.137*y**2 - 0.070*y**3
        # bf = 13.670 + 4.257*y - 0.420*y**2 + 0.374*y**3

        # Final result
        # a = (ai*(x<xi1) + av*((x>xi1) & (x<xi3))
        #         + au*((x>xi3)&(x<8.0)) + af*(x>8.0))
        # b = (bi*(x<xi1) + bv*((x>xi1) & (x<xi3))
        #         + bu*((x>xi3)&(x<8.0)) + bf*(x>8.0))

        # Final result
        a = ai * (x < xi1) + av * ((x >= xi1) & (x < xi3))
        b = bi * (x < xi1) + bv * ((x >= xi1) & (x < xi3))

        return a + b / Rv


class G16(BaseExtRvAfAModel):
    r"""
    Gordon et al (2016) Milky Way, LMC, & SMC R(V) and f_A dependent model

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

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

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

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_G16, "G16")

        # just in case someone calls evaluate explicitly
        RvA = np.atleast_1d(RvA)

        # ensure Rv is a single element, not numpy array
        RvA = RvA[0]

        # get the A component extinction model
        extA_model = F99(Rv=RvA)
        alav_A = extA_model(x / u.micron)

        # get the B component extinction model
        extB_model = G03_SMCBar()
        alav_B = extB_model(x / u.micron)

        # create the mixture model
        alav = fA * alav_A + (1.0 - fA) * alav_B

        # return A(x)/A(V)
        return alav


class F19(BaseExtRvModel):
    r"""
    Fitzpatrick et al (2019) extinction model calculation

    Fitzpatrick, Massa, Gordon et al. (2019, ApJ, 886, 108) model.
    Based on a sample of stars observed spectroscopically in the
    optical with HST/STIS.

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
    F19 Milky Way R(V) dependent extinction model

    Example showing F19 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import F19

        fig, ax = plt.subplots()

        # temp model to get the correct x range
        text_model = F19()

        # generate the curves and plot them
        x = np.arange(text_model.x_range[0],
                      text_model.x_range[1],0.1)/u.micron

        Rvs = [2.0, 3.0, 4.0, 5.0, 6.0]
        for cur_Rv in Rvs:
           ext_model = F19(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    Rv_range = [2.0, 6.0]
    x_range = [0.3, 8.7]

    def __init__(self, Rv=3.1, **kwargs):

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(data_path + "F19_tabulated.dat", format="ascii")

        # compute E(lambda-55)/E(B-55) on the tabulated x points
        self.k_rV_tab_x = a["k_3.02"].data + a["deltak"].data * (Rv - 3.10) * 0.990

        # setup spline interpolation
        self.spline_rep = interpolate.splrep(a["x"].data, self.k_rV_tab_x)

        super().__init__(Rv, **kwargs)

    def evaluate(self, in_x, Rv):
        """
        F19 function

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, self.__class__.__name__)

        # just in case someone calls evaluate explicitly
        Rv = np.atleast_1d(Rv)

        # ensure Rv is a single element, not numpy array
        Rv = Rv[0]

        # use spline interpolation to evaluate the curve for the input x values
        k_rV = interpolate.splev(x, self.spline_rep, der=0)

        # convert to A(x)/A(55) from E(x-55)/E(44-55)
        a_rV = k_rV / Rv + 1.0

        # return A(x)/A(55)
        return a_rV


class D22(BaseExtRvModel):
    r"""
    Decleir et al (2022) extinction model calculation

    Decleir, Gordon, et al. (2022, ApJ, submitted) model.
    Based on a sample of stars observed spectroscopically in the
    NIR with IRTF/SpeX.

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
    D22 Milky Way R(V) dependent extinction model

    Example showing D22 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import D22

        fig, ax = plt.subplots()

        # temp model to get the correct x range
        text_model = D22()

        # generate the curves and plot them
        x = np.arange(text_model.x_range[0], text_model.x_range[1], 0.01) / u.micron

        Rvs = [2.5, 3.1, 4.0, 4.75, 5.5]
        for cur_Rv in Rvs:
            ext_model = D22(Rv=cur_Rv)
            ax.plot(1. / x, ext_model(x), label="R(V) = " + str(cur_Rv))

        ax.set_xlabel(r"$\lambda$ [$\mu m$]")
        ax.set_ylabel(r"$A(x)/A(V)$")

        ax.legend(loc="best")
        plt.show()
    """

    Rv_range = [2.5, 5.5]
    x_range = [1 / 4.0, 1 / 0.80]

    def __init__(self, Rv=3.1, **kwargs):

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(data_path + "D22_Rv_slope.dat", format="ascii")

        # setup spline interpolation
        self.spline_rep = interpolate.splrep(a["wavelength[micron]"].data, a["slope"])

        super().__init__(Rv, **kwargs)

    def evaluate(self, in_x, Rv):
        """
        D22 function

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, self.__class__.__name__)

        # just in case someone calls evaluate explicitly
        Rv = np.atleast_1d(Rv)

        # ensure Rv is a single element, not numpy array
        Rv = Rv[0]

        # intercepts
        mod_a = PowerLaw1D(amplitude=0.377, alpha=1.78, x_0=1.0)
        a = mod_a(1.0 / x)

        # slopes
        # from spline interpolation
        b = interpolate.splev(1.0 / x, self.spline_rep, der=0)

        # return A(x)/A(V)
        return a + b * (1.0 / Rv - 1 / 3.1)


class G23(BaseExtRvModel):
    r"""
    Gordon et al. (2023) Milky Way R(V) dependent model

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
    From Gordon et al. (2023, ApJ, in press)

    Example showing CCM89 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import G23

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(0.0912), np.log10(30.0), num=1000) * u.micron

        Rvs = [2.5, 3.1, 4.0, 4.75, 5.5]
        for cur_Rv in Rvs:
           ext_model = G23(Rv=cur_Rv)
           ax.plot(lam,ext_model(lam),label='R(V) = ' + str(cur_Rv))

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('$\lambda$ [$\mu$m]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    Rv_range = [2.3, 5.6]
    x_range = x_range_G23

    def evaluate(self, in_x, Rv):
        """
        G23 function

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
        x = _get_x_in_wavenumbers(in_x)

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, self.x_range, "G23")

        # setup the a & b coefficient vectors
        n_x = len(x)
        self.a = np.zeros(n_x)
        self.b = np.zeros(n_x)

        # define the ranges
        ir_indxs = np.where(np.logical_and(1.0 / 35.0 <= x, x < 1.0 / 1.0))
        opt_indxs = np.where(np.logical_and(1.0 / 1.1 <= x, x < 1.0 / 0.3))
        uv_indxs = np.where(np.logical_and(1.0 / 0.3 <= x, x <= 1.0 / 0.09))

        # overlap ranges
        optir_waves = [0.9, 1.1]
        optir_overlap = (x >= 1.0 / optir_waves[1]) & (x <= 1.0 / optir_waves[0])
        uvopt_waves = [0.3, 0.33]
        uvopt_overlap = (x >= 1.0 / uvopt_waves[1]) & (x <= 1.0 / uvopt_waves[0])

        # NIR/MIR
        # fmt: off
        # (scale, alpha1, alpha2, swave, swidth), sil1, sil2
        ir_a = [0.38526, 1.68467, 0.78791, 4.30578, 4.78338,
                0.06652, 9.8434, 2.21205, -0.24703,
                0.0267 , 19.58294, 17., -0.27]
        # fmt: on
        ir_b = [-1.01251, 1.0, -1.06099]
        self.a[ir_indxs] = self.nirmir_intercept(x[ir_indxs], ir_a)

        irpow = PowerLaw1D()
        irpow.parameters = ir_b
        self.b[ir_indxs] = irpow(x[ir_indxs])

        # optical
        # fmt: off
        # polynomial coeffs, ISS1, ISS2, ISS3
        opt_a = [-0.35848, 0.7122 , 0.08746, -0.05403, 0.00674,
                 0.03893, 2.288, 0.243,
                 0.02965, 2.054, 0.179,
                 0.01747, 1.587, 0.243]
        opt_b = [0.12354, -2.68335, 2.01901, -0.39299, 0.03355,
                 0.18453, 2.288, 0.243,
                 0.19728, 2.054, 0.179,
                 0.1713 , 1.587, 0.243]
        # fmt: on
        m20_model_a = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_a.parameters = opt_a
        self.a[opt_indxs] = m20_model_a(x[opt_indxs])
        m20_model_b = Polynomial1D(4) + Drude1D() + Drude1D() + Drude1D()
        m20_model_b.parameters = opt_b
        self.b[opt_indxs] = m20_model_b(x[opt_indxs])

        # overlap between optical/ir
        # weights = (1.0 / optir_waves[1] - x[optir_overlap]) / (
        #     1.0 / optir_waves[1] - 1.0 / optir_waves[0]
        # )
        weights = _smoothstep(
            1.0 / x[optir_overlap], x_min=optir_waves[0], x_max=optir_waves[1], N=1
        )
        self.a[optir_overlap] = (1.0 - weights) * m20_model_a(x[optir_overlap])
        self.a[optir_overlap] += weights * self.nirmir_intercept(
            x[optir_overlap], ir_a
        )
        self.b[optir_overlap] = (1.0 - weights) * m20_model_b(x[optir_overlap])
        self.b[optir_overlap] += weights * irpow(x[optir_overlap])

        # Ultraviolet
        uv_a = [0.81297, 0.2775, 1.06295, 0.11303, 4.60, 0.99]
        uv_b = [-2.97868, 1.89808, 3.10334, 0.65484, 4.60, 0.99]

        fm90_model_a = FM90()
        fm90_model_a.parameters = uv_a
        self.a[uv_indxs] = fm90_model_a(x[uv_indxs] / u.micron)
        fm90_model_b = FM90()
        fm90_model_b.parameters = uv_b
        self.b[uv_indxs] = fm90_model_b(x[uv_indxs] / u.micron)

        # overlap between uv/optical
        # weights = (1.0 / uvopt_waves[1] - x[uvopt_overlap]) / (
        #     1.0 / uvopt_waves[1] - 1.0 / uvopt_waves[0]
        # )
        weights = _smoothstep(
            1.0 / x[uvopt_overlap], x_min=uvopt_waves[0], x_max=uvopt_waves[1], N=1
        )
        self.a[uvopt_overlap] = (1.0 - weights) * fm90_model_a(x[uvopt_overlap] / u.micron)
        self.a[uvopt_overlap] += weights * m20_model_a(x[uvopt_overlap])
        self.b[uvopt_overlap] = (1.0 - weights) * fm90_model_b(x[uvopt_overlap] / u.micron)
        self.b[uvopt_overlap] += weights * m20_model_b(x[uvopt_overlap])

        # return A(x)/A(V)
        return self.a + self.b * (1 / Rv - 1 / 3.1)

    @staticmethod
    def nirmir_intercept(x, params):
        """
        Functional form for the NIR/MIR intercept term.
        Based on modifying the G21 shape model to have two power laws instead
        of one with a break wavelength.

        Parameters
        ----------
        x: float
           expects x in wavenumbers [1/micron]
        params: floats
           paramters of function

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]
        """
        wave = 1 / x

        # fmt: off
        (scale, alpha, alpha2, swave, swidth,
            sil1_amp, sil1_center, sil1_fwhm, sil1_asym,
            sil2_amp, sil2_center, sil2_fwhm, sil2_asym) = params
        # fmt: on

        # broken powerlaw with a smooth transition
        axav_pow1 = scale * (wave ** (-1.0 * alpha))

        norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
        axav_pow2 = scale * norm_ratio * (wave ** (-1.0 * alpha2))

        # use smoothstep to smoothly transition between the two powerlaws
        weights = _smoothstep(
            wave, x_min=swave - swidth / 2, x_max=swave + swidth / 2, N=1
        )
        axav = axav_pow1 * (1.0 - weights) + axav_pow2 * weights

        # silicate feature drudes
        axav += _modified_drude(wave, sil1_amp, sil1_center, sil1_fwhm, sil1_asym)
        axav += _modified_drude(wave, sil2_amp, sil2_center, sil2_fwhm, sil2_asym)

        return axav
