import importlib.resources as importlib_resources

import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table
from astropy.modeling.models import PowerLaw1D

from .baseclasses import BaseExtModel
from .shapes import P92, G21, _curve_F99_method

__all__ = [
    "RL85_MWGC",
    "RRP89_MWGC",
    "B92_MWAvg",
    "G03_SMCBar",
    "G03_LMCAvg",
    "G03_LMC2",
    "I05_MWAvg",
    "CT06_MWGC",
    "CT06_MWLoc",
    "GCC09_MWAvg",
    "F11_MWGC",
    "G21_MWAvg",
    "D22_MWAvg",
    "G24_SMCAvg",
    "G24_SMCBumps",
    "C25_M31Avg",
    "G25_M33Avg",
]


class RL85_MWGC(BaseExtModel):
    r"""
    Reike & Lebofsky (1985) MW Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Rieke & Lebofsky (1985, ApJ,288, 618)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import RL85_MWGC

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = RL85_MWGC()

        # generate the curves and plot them
        x = np.arange(1.0/ext_model.x_range[1], 1.0/ext_model.x_range[0], 0.1) * u.micron

        ax.plot(x,ext_model(x),label='RL85_MWGC')
        ax.plot(1.0/ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 13.0, 1.0 / 1.25]

    Rv = 3.09

    # fmt: off
    obsdata_x = 1.0 / np.array(
        [13.0, 12.5, 12.0, 11.5, 11.0, 10.5, 10.0,
         9.5, 9.0, 8.5, 8.0, 4.8, 3.5, 2.22, 1.65, 1.25]
    )
    obsdata_axav = np.array(
        [0.027, 0.030, 0.037, 0.047, 0.060, 0.074, 0.083,
         0.087, 0.074, 0.043, 0.020, 0.023, 0.058, 0.112, 0.175, 0.282]
    )
    # fmt: on

    # accuracy of the observed data based on published table
    obsdata_tolerance = 1e-6

    @classmethod
    def evaluate(cls, x):
        r"""
        RL85 MWGC function

        Parameters
        ----------
        x: float
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
        # define the function using simple linear interpolation
        # avoids negative values of alav that happens with cubic splines
        f = interp1d(cls.obsdata_x, cls.obsdata_axav)

        return f(x)


class RRP89_MWGC(BaseExtModel):
    r"""
    Reike, Rieke, & Paul (1989) MW Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Rieke, Rieke, & Paul (1989, ApJ, 336, 752)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import RRP89_MWGC

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = RRP89_MWGC()

        # generate the curves and plot them
        x = np.arange(1.0/ext_model.x_range[1], 1.0/ext_model.x_range[0], 0.1) * u.micron

        ax.plot(x,ext_model(x),label='RRP89_MWGC')
        ax.plot(1.0/ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 13.0, 1.0 / 0.90]

    Rv = 3.09

    # fmt: off
    obsdata_x = 1.0 / np.array(
        [0.90, 1.25, 1.6, 2.2, 3.5, 4.8, 8.0,
         9.5, 10.6, 11.0, 13.0]
    )
    obsdata_elvebv = np.array(
        [-1.60, -2.22, -2.55, -2.744, -2.88, -2.99, -3.01,
         -2.73, -2.87, -2.84, -2.98]
    )
    # fmt: on
    obsdata_axav = obsdata_elvebv / Rv + 1.0

    # accuracy of the observed data based on published table
    obsdata_tolerance = 1e-6

    @classmethod
    def evaluate(cls, x):
        r"""
        RRP89 MWGC function

        Parameters
        ----------
        x: float
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
        # define the function using simple linear interpolation
        # avoids negative values of alav that happens with cubic splines
        f = interp1d(cls.obsdata_x, cls.obsdata_axav)

        return f(x)


class B92_MWAvg(BaseExtModel):
    r"""
    Bastiaansen (1992) Optical Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Bastiaansen (1992, A&AS, 93, 449-462)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import B92_MWAvg

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = B92_MWAvg()

        # generate the curves and plot them
        x = np.arange(1.0/ext_model.x_range[1], 1.0/ext_model.x_range[0], 0.1) * u.micron

        ax.plot(x,ext_model(x),label='B92')
        ax.plot(1.0/ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 0.7873, 1.0 / 0.3402]

    Rv = 3.1  # assumed!

    # fmt: off
    obsdata_x = 1.0 / np.array(
        [0.7873, 0.7505, 0.7102, 0.6681, 0.64,
         0.6107, 0.5821, 0.5601, 0.5407, 0.5205,
         0.4999, 0.4708, 0.4496, 0.4395, 0.4192,
         0.4038, 0.3785, 0.36, 0.3493, 0.3402]
    )
    obsdata_axav = np.array(
        [0.849, 0.891, 0.941, 0.998, 1.045, 1.088,
         1.139, 1.176, 1.226, 1.279, 1.34 , 1.418,
         1.473, 1.507, 1.556, 1.595, 1.659, 1.718,
         1.761, 1.795]
    )
    # fmt: on

    # accuracy of the observed data based on published table
    obsdata_tolerance = 6e-3

    @classmethod
    def evaluate(cls, x):
        """
        B92 function

        Parameters
        ----------
        x: float
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
        # define the function allowing for spline interpolation
        f = interp1d(cls.obsdata_x, cls.obsdata_axav)

        return f(x)


class G03_SMCBar(BaseExtModel):
    r"""
    Gordon et al (2003) SMCBar Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Gordon et al. (2003, ApJ, 594, 279)

    The observed A(lambda)/A(V) values at 2.198 and 1.25 microns were
    changed to provide smooth interpolation as noted in
    Gordon et al. (2016, ApJ, 826, 104)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import G03_SMCBar

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = G03_SMCBar()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='G03 SMCBar')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 10.0]

    Rv = 2.74

    # fmt: off
    obsdata_x = np.array(
        [0.455, 0.606, 0.800, 1.235, 1.538, 1.818, 2.273, 2.703,
         3.375, 3.625, 3.875, 4.125, 4.375, 4.625, 4.875, 5.125,
         5.375, 5.625, 5.875, 6.125, 6.375, 6.625, 6.875, 7.125,
         7.375, 7.625, 7.875, 8.125, 8.375, 8.625]
    )
    obsdata_axav = np.array(
        [0.110, 0.169, 0.250, 0.567, 0.801, 1.000, 1.374, 1.672,
         2.000, 2.220, 2.428, 2.661, 2.947, 3.161, 3.293, 3.489,
         3.637, 3.866, 4.013, 4.243, 4.472, 4.776, 5.000, 5.272,
         5.575, 5.795, 6.074, 6.297, 6.436, 6.992]
    )
    # fmt: on

    # accuracy of the observed data based on published table
    obsdata_tolerance = 6e-2

    def evaluate(self, x):
        """
        G03 SMCBar function

        Parameters
        ----------
        x: float
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
        C1 = -4.959
        C2 = 2.264
        C3 = 0.389
        C4 = 0.461
        xo = 4.6
        gamma = 1.0

        optnir_axav_x = 1.0 / np.array(
            [2.198, 1.65, 1.25, 0.81, 0.65, 0.55, 0.44, 0.37]
        )
        # values at 2.198 and 1.25 changed to provide smooth interpolation
        # as noted in Gordon et al. (2016, ApJ, 826, 104)
        optnir_axav_y = [0.11, 0.169, 0.25, 0.567, 0.801, 1.00, 1.374, 1.672]

        # return A(x)/A(V)
        return _curve_F99_method(
            x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
        )


class G03_LMCAvg(BaseExtModel):
    r"""
    Gordon et al (2003) LMCAvg Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Gordon et al. (2003, ApJ, 594, 279)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import G03_LMCAvg

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = G03_LMCAvg()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='G03 LMCAvg')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 10.0]

    Rv = 3.41

    # fmt: off
    obsdata_x = np.array(
        [0.455, 0.606, 0.800, 1.818, 2.273, 2.703, 3.375, 3.625,
         3.875, 4.125, 4.375, 4.625, 4.875, 5.125, 5.375, 5.625,
         5.875, 6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625,
         7.875, 8.125]
    )
    obsdata_axav = np.array(
        [0.100, 0.186, 0.257, 1.000, 1.293, 1.518, 1.786, 1.969,
         2.149, 2.391, 2.771, 2.967, 2.846, 2.646, 2.565, 2.566,
         2.598, 2.607, 2.668, 2.787, 2.874, 2.983, 3.118, 3.231,
         3.374, 3.366]
    )
    # fmt: on

    # accuracy of the observed data based on published table
    obsdata_tolerance = 6e-2

    def evaluate(self, x):
        """
        G03 LMCAvg function

        Parameters
        ----------
        x: float
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
        C1 = -0.890
        C2 = 0.998
        C3 = 2.719
        C4 = 0.400
        xo = 4.579
        gamma = 0.934

        optnir_axav_x = 1.0 / np.array([2.198, 1.65, 1.25, 0.55, 0.44, 0.37])
        # value at 2.198 changed to provide smooth interpolation
        # as noted in Gordon et al. (2016, ApJ, 826, 104) for SMCBar
        optnir_axav_y = [0.10, 0.186, 0.257, 1.000, 1.293, 1.518]

        # return A(x)/A(V)
        return _curve_F99_method(
            x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
        )


class G03_LMC2(BaseExtModel):
    r"""
    Gordon et al (2003) LMC2 Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Gordon et al. (2003, ApJ, 594, 279)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import G03_LMC2

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.3,10.0,0.1)/u.micron

        # define the extinction model
        ext_model = G03_LMC2()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='G03 LMC2')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 10.0]

    Rv = 2.76

    # fmt: off
    obsdata_x = np.array(
        [0.455, 0.606, 0.800, 1.818, 2.273, 2.703, 3.375, 3.625,
         3.875, 4.125, 4.375, 4.625, 4.875, 5.125, 5.375, 5.625,
         5.875, 6.125, 6.375, 6.625, 6.875, 7.125, 7.375, 7.625,
         7.875, 8.125]
    )
    obsdata_axav = np.array(
        [0.101, 0.150, 0.299, 1.000, 1.349, 1.665, 1.899, 2.067,
         2.249, 2.447, 2.777, 2.922, 2.921, 2.812, 2.805, 2.863,
         2.932, 3.060, 3.110, 3.299, 3.408, 3.515, 3.670, 3.862,
         3.937, 4.055]
    )
    # fmt: on

    # accuracy of the observed data based on published table
    obsdata_tolerance = 6e-2

    def evaluate(self, x):
        """
        G03 LMC2 function

        Parameters
        ----------
        x: float
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
        C1 = -1.475
        C2 = 1.132
        C3 = 1.463
        C4 = 0.294
        xo = 4.558
        gamma = 0.945

        optnir_axav_x = 1.0 / np.array([2.198, 1.65, 1.25, 0.55, 0.44, 0.37])
        # value at 1.65 changed to provide smooth interpolation
        # as noted in Gordon et al. (2016, ApJ, 826, 104) for SMCBar
        optnir_axav_y = [0.101, 0.15, 0.299, 1.000, 1.349, 1.665]

        # return A(x)/A(V)
        return _curve_F99_method(
            x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
        )


class I05_MWAvg(BaseExtModel):
    r"""
    Indebetouw et al (2005) MW Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Indebetouw et al. (2005, ApJ, 619, 931)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import I05_MWAvg

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = I05_MWAvg()

        # generate the curves and plot them
        x = np.arange(1.0/ext_model.x_range[1], 1.0/ext_model.x_range[0], 0.1) * u.micron

        ax.plot(x,ext_model(x),label='I05_MWAvg')
        ax.plot(1.0/ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 7.76, 1.0 / 1.24]

    Rv = 3.1  # assumed!

    # fmt: off
    obsdata_x = 1.0 / np.array(
        [1.24, 1.664, 2.164, 3.545, 4.442, 5.675, 7.760]
    )
    obsdata_axav = np.array(
        [2.50, 1.55, 1.00, 0.56, 0.43, 0.43, 0.43]
    ) * 0.112  # ak/av = 0.112 (F19, Rv = 3.1)

    obsdata_axav_unc = np.array(
        [0.15, 0.08, 0.0, 0.06, 0.08, 0.10, 0.10]
    ) * 0.112  # ak/av = 0.112 (F19, Rv = 3.1)
    # fmt: on

    # accuracy of the observed data based on published table
    obsdata_tolerance = 1e-6

    def evaluate(self, x):
        """
        I05 MWAvg function

        Parameters
        ----------
        x: float
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
        # define the function allowing for spline interpolation
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)


class CT06_MWGC(BaseExtModel):
    r"""
    Chiar & Tielens (2006) MW Galactic Center Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Chiar & Tielens (2006, ApJ, 637 774)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import CT06_MWGC

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = CT06_MWGC()

        # generate the curves and plot them
        x = np.arange(1.0/ext_model.x_range[1], 1.0/ext_model.x_range[0], 0.1) * u.micron

        ax.plot(x,ext_model(x),label='CT06_MWGC')
        ax.plot(1.0/ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 27.0, 1.0 / 1.24]

    Rv = 3.1  # assumed!

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            a = Table.read(
                data_path / "CT06_pixiedust.dat", format="ascii.commented_header"
            )

        self.obsdata_x = 1.0 / a["wave"].data
        # ext is A(lambda)/A(K)
        # A(K)/A(V) = 0.112 (F19, R(V) = 3.1)
        self.obsdata_axav = 0.112 * a["galcen"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 1e-6

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        CT06 MWGC function

        Parameters
        ----------
        x: float
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
        # define the function allowing for spline interpolation
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)


class CT06_MWLoc(BaseExtModel):
    r"""
    Chiar & Tielens (2006) MW Local ISM Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Chiar & Tielens (2006, ApJ, 637 774)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import CT06_MWLoc

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = CT06_MWLoc()

        # generate the curves and plot them
        x = np.arange(1.0/ext_model.x_range[1], 1.0/ext_model.x_range[0], 0.1) * u.micron

        ax.plot(x,ext_model(x),label='CT06_MWLoc')
        ax.plot(1.0/ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 27.0, 1.0 / 1.24]

    Rv = 3.1  # assumed!

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            a = Table.read(
                data_path / "CT06_pixiedust.dat", format="ascii.commented_header"
            )

        self.obsdata_x = 1.0 / a["wave"].data
        # ext is A(lambda)/A(K)
        # A(K)/A(V) = 0.112 (F19, R(V) = 3.1)
        self.obsdata_axav = 0.112 * a["local"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 1e-6

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        CG06 MWLoc function

        Parameters
        ----------
        x: float
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
        # define the function allowing for spline interpolation
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)


class GCC09_MWAvg(BaseExtModel):
    r"""
    Gordon, Cartledge, & Clayton (2009) Milky Way Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Gordon, Cartledge, & Clayton (2009, ApJ, 705, 1320)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import GCC09_MWAvg

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.3,1.0/0.0912,0.1)/u.micron

        # define the extinction model
        ext_model = GCC09_MWAvg()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='GCC09_MWAvg')
        ax.errorbar(ext_model.obsdata_x_fuse, ext_model.obsdata_axav_fuse,
                    yerr=ext_model.obsdata_axav_unc_fuse,
                    fmt='ko', label='obsdata (FUSE)')
        ax.errorbar(ext_model.obsdata_x_iue, ext_model.obsdata_axav_iue,
                    yerr=ext_model.obsdata_axav_unc_iue,
                    fmt='bs', label='obsdata (IUE)')
        ax.errorbar(ext_model.obsdata_x_bands, ext_model.obsdata_axav_bands,
                    yerr=ext_model.obsdata_axav_unc_bands,
                    fmt='g^', label='obsdata (Opt/NIR)')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.09, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 1.0 / 0.0912]

    Rv = 3.1

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            # GCC09 sigma clipped average of 75 sightlines
            a = Table.read(
                data_path / "GCC09_FUSE.dat", format="ascii.commented_header"
            )
            b = Table.read(data_path / "GCC09_IUE.dat", format="ascii.commented_header")
            c = Table.read(
                data_path / "GCC09_PHOT.dat", format="ascii.commented_header"
            )

        # FUSE range
        self.obsdata_x_fuse = a["x"].data
        self.obsdata_axav_fuse = a["ext"].data
        self.obsdata_axav_unc_fuse = a["unc"].data

        # IUE range
        self.obsdata_x_iue = b["x"].data
        self.obsdata_axav_iue = b["ext"].data
        self.obsdata_axav_unc_iue = b["unc"].data

        # Opt/NIR range
        self.obsdata_x_bands = c["x"].data
        self.obsdata_axav_bands = c["ext"].data
        self.obsdata_axav_unc_bands = c["unc"].data

        # put them together
        self.obsdata_x = np.concatenate(
            (self.obsdata_x_fuse, self.obsdata_x_iue, self.obsdata_x_bands)
        )
        self.obsdata_axav = np.concatenate(
            (self.obsdata_axav_fuse, self.obsdata_axav_iue, self.obsdata_axav_bands)
        )
        self.obsdata_axav_unc = np.concatenate(
            (
                self.obsdata_axav_unc_fuse,
                self.obsdata_axav_unc_iue,
                self.obsdata_axav_unc_bands,
            )
        )

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 5e-1

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        GCC09_MWAvg function

        Parameters
        ----------
        x: float
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
        # P92 parameters fit to the data using uncs as weights
        p92_fit = P92(
            BKG_amp=203.805939127,
            BKG_lambda=0.0508199427208,
            BKG_b=88.0591826413,
            BKG_n=2.0,
            FUV_amp=5.33962141873,
            FUV_lambda=0.08,
            FUV_b=-0.777129536415,
            FUV_n=3.88322376926,
            NUV_amp=0.0447023090042,
            NUV_lambda=0.217548391182,
            NUV_b=-1.95723797612,
            NUV_n=2.0,
            SIL1_amp=0.00264935064935,
            SIL1_lambda=9.7,
            SIL1_b=-1.95,
            SIL1_n=2.0,
            SIL2_amp=0.00264935064935,
            SIL2_lambda=18.0,
            SIL2_b=-1.80,
            SIL2_n=2.0,
            FIR_amp=0.01589610389,
            FIR_lambda=25.0,
            FIR_b=0.0,
            FIR_n=2.0,
        )

        # return A(x)/A(V)
        return p92_fit(x)


class F11_MWGC(BaseExtModel):
    r"""
    Fritz et al (2011) MW Galactic Center Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Fritz et al. (2011, ApJ, 737, 73)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import F11_MWGC

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = F11_MWGC()

        # generate the curves and plot them
        x = np.arange(1.0/ext_model.x_range[1], 1.0/ext_model.x_range[0], 0.1) * u.micron

        ax.plot(x,ext_model(x),label='F11_MWGC')
        ax.plot(1.0/ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 19.062, 1.0 / 1.282]

    Rv = 3.1  # assumed!

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            a = Table.read(
                data_path / "fritz11_galcenter.dat", format="ascii.commented_header"
            )

        self.obsdata_x = 1.0 / a["wave"].data
        # ext is total extinction to GalCenter
        # A(K) = 2.42
        # A(K)/A(V) = 0.112 (F19, R(V) = 3.1)
        self.obsdata_axav = 0.112 * a["ext"].data / 2.42
        self.obsdata_axav_unc = 0.112 * a["unc"].data / 2.42

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 1e-6

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        F11 MWGC function

        Parameters
        ----------
        x: float
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
        # define the function allowing for spline interpolation
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)


class G21_MWAvg(BaseExtModel):
    r"""
    Gordon et al. (2021) Milky Way Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Gordon et al. (2021, ApJ, 916, 33)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import G21_MWAvg

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(1.01), np.log10(31.9), num=1000)
        x = (1.0/lam)/u.micron

        # define the extinction model
        ext_model = G21_MWAvg()

        ax.plot(1.0/x,ext_model(x),label='G21_MWAvg')
        ax.errorbar(1.0/ext_model.obsdata_x_irs, ext_model.obsdata_axav_irs,
                    yerr=ext_model.obsdata_axav_unc_irs,
                    fmt='ko', label='obsdata (IRS)')
        ax.errorbar(1.0/ext_model.obsdata_x_bands, ext_model.obsdata_axav_bands,
                    yerr=ext_model.obsdata_axav_unc_bands,
                    fmt='g^', label='obsdata (photometry)')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 32.0, 1.0]

    Rv = 3.17

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            # GCC09 sigma clipped average of 75 sightlines
            a = Table.read(data_path / "G21_IRS.dat", format="ascii.commented_header")
            b = Table.read(data_path / "G21_PHOT.dat", format="ascii.commented_header")

        # IRS range
        self.obsdata_x_irs = 1.0 / a["wave"].data
        self.obsdata_axav_irs = a["ext"].data
        self.obsdata_axav_unc_irs = a["unc"].data

        # Opt/NIR range
        self.obsdata_x_bands = 1.0 / b["wave"].data
        self.obsdata_axav_bands = b["ext"].data
        self.obsdata_axav_unc_bands = b["unc"].data

        # put them together
        self.obsdata_x = np.concatenate((self.obsdata_x_irs, self.obsdata_x_bands))
        self.obsdata_axav = np.concatenate(
            (self.obsdata_axav_irs, self.obsdata_axav_bands)
        )
        self.obsdata_axav_unc = np.concatenate(
            (
                self.obsdata_axav_unc_irs,
                self.obsdata_axav_unc_bands,
            )
        )

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 5e-1

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        G21_MWAvg function

        Parameters
        ----------
        x: float
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
        # G21 parameters fit to the data using uncs as weights
        g21_fit = G21(
            scale=0.366,
            alpha=1.480,
            sil1_amp=0.06893,
            sil1_center=9.865,
            sil1_fwhm=2.507,
            sil1_asym=-0.232,
            sil2_amp=0.02684,
            sil2_center=19.973,
            sil2_fwhm=16.989,
            sil2_asym=-0.273,
        )

        # return A(x)/A(V)
        # G21 a full dust_extinction model, hence send in x with units
        return g21_fit(x)


class D22_MWAvg(BaseExtModel):
    r"""
    Decleir et al. (2022) Milky Way Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Decleir et al. (2022, ApJ, 930, 15)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import D22_MWAvg

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(0.8), np.log10(4.9), num=1000)
        x = (1.0 / lam) / u.micron

        # define the extinction model
        ext_model = D22_MWAvg()

        ax.plot(1.0 / x, ext_model(x), label="D22_MWAvg")
        ax.errorbar(
            1.0 / ext_model.obsdata_x,
            ext_model.obsdata_axav,
            yerr=ext_model.obsdata_axav_unc,
            fmt="ko",
            label="obsdata",
        )

        ax.set_xlabel(r"$\lambda$ [$\mu m$]")
        ax.set_ylabel(r"$A(x)/A(V)$")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.legend(loc="best")
        plt.show()

    """

    x_range = [1.0 / 5.0, 1.0 / 0.8]

    Rv = 3.12

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            # D22 sigma clipped average of 13 diffuse sightlines
            a = Table.read(data_path / "D22.dat", format="ascii.commented_header")

        # Spex data
        self.obsdata_x = 1.0 / a["wavelength[micron]"].data
        self.obsdata_axav = a["ave"].data
        self.obsdata_axav_unc = a["ave_unc"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 0.2  # check

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        D22_MWAvg function

        Parameters
        ----------
        x: float
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
        # setup the model
        d22_fit = PowerLaw1D(alpha=1.71, amplitude=0.386, x_0=1.0)

        # return A(x)/A(V)
        # Note that model in D22 was done versus wavelength in microns
        return d22_fit(1.0 / x)


class G24_SMCAvg(BaseExtModel):
    r"""
    Gordon et al (2024) SMC Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Gordon et al. (2024, ApJ, 970, 51)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import G24_SMCAvg

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = G24_SMCAvg()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='G24 SMC Average')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 10.0]

    Rv = 3.02

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            a = Table.read(
                data_path / "G24_SMCAvg.dat", format="ascii.commented_header"
            )

        # data
        self.obsdata_x = 1.0 / a["wave"].data
        self.obsdata_axav = a["A(l)/A(V)"].data
        self.obsdata_axav_unc = a["unc"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = (
            0.1  # large value driven by short wavelength uncertainties
        )

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        G24 SMCAvg function

        Parameters
        ----------
        x: float
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
        C1 = -5.07
        C2 = 2.30
        C3 = 0.12
        C4 = 0.07
        xo = 4.59
        gamma = 0.95

        optnir_axav_x = 1.0 / np.array([2.198, 1.65, 1.25, 0.55, 0.44, 0.37])
        optnir_axav_y = [0.062, 0.125, 0.324, 1.021, 1.349, 1.514]

        # return A(x)/A(V)
        return _curve_F99_method(
            x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
        )


class G24_SMCBumps(BaseExtModel):
    r"""
    Gordon et al (2024) SMC Bumps Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Gordon et al. (2024, ApJ, 970, 51)

    Two data points in the FUV from the data file giving the observed average were removed
    as they are *very* deviate from the FM90 parametrization.  This cause the automated tests
    to fail.

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import G24_SMCBumps

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = G24_SMCBumps()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='G24 SMC Bumps')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 10.0]

    Rv = 2.55

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            a = Table.read(
                data_path / "G24_SMCBumps.dat", format="ascii.commented_header"
            )

        # data
        self.obsdata_x = 1.0 / a["wave"].data
        self.obsdata_axav = a["A(l)/A(V)"].data
        self.obsdata_axav_unc = a["unc"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = (
            0.1  # large value driven by short wavelength uncertainties
        )

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        G24 SMCBumps function

        Parameters
        ----------
        x: float
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
        C1 = -2.85
        C2 = 1.51
        C3 = 2.64
        C4 = 0.25
        xo = 4.73
        gamma = 1.15

        optnir_axav_x = 1.0 / np.array([2.198, 1.65, 1.25, 0.55, 0.44, 0.37])
        optnir_axav_y = [0.055, 0.162, 0.293, 1.017, 1.400, 1.684]

        # return A(x)/A(V)
        return _curve_F99_method(
            x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
        )


class C25_M31Avg(BaseExtModel):
    r"""
    Clayton et al (2025) M31 Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Clayton et al. (2025, ApJ, in press)

    One data point in the FUV from the data file giving the observed average was removed
    as it is *very* deviate from the FM90 parametrization.  This cause the automated tests
    to fail.

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import C25_M31Avg

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = C25_M31Avg()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='C25 M31 Average')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 10.0]

    Rv = 3.20

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            # D22 sigma clipped average of 13 diffuse sightlines
            a = Table.read(
                data_path / "C25_M31Ave.dat", format="ascii.commented_header"
            )

        # data
        self.obsdata_x = 1.0 / a["wave"].data
        self.obsdata_axav = a["A(l)/A(V)"].data
        self.obsdata_axav_unc = a["unc"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = (
            0.1  # large value driven by short wavelength uncertainties
        )

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        C25 M31Avg function

        Parameters
        ----------
        x: float
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
        C1 = -1.57
        C2 = 1.21
        B3 = 3.00
        C4 = 0.13
        xo = 4.623
        gamma = 1.04
        C3 = B3 * (gamma**2)

        optnir_axav_x = 1.0 / np.array([1.537, 1.153, 0.805, 0.475, 0.335])
        optnir_axav_y = [0.143, 0.331, 0.545, 1.205, 1.703]

        # return A(x)/A(V)
        return _curve_F99_method(
            x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
        )


class G25_M33Avg(BaseExtModel):
    r"""
    Gordon et al (2025) M33 Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Gordon et al. (2025, ApJ, submitted)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.averages import G25_M33Avg

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = G25_M33Avg()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='G25 M33 Average')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 10.0]

    Rv = 4.66

    def __init__(self, **kwargs):

        # get the tabulated information
        ref = importlib_resources.files("dust_extinction") / "data"
        with importlib_resources.as_file(ref) as data_path:
            # D22 sigma clipped average of 13 diffuse sightlines
            a = Table.read(
                data_path / "G25_M33Ave.dat", format="ascii.commented_header"
            )

        # data
        self.obsdata_x = 1.0 / a["wave"].data
        self.obsdata_axav = a["A(l)/A(V)"].data
        self.obsdata_axav_unc = a["unc"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = (
            0.1  # large value driven by short wavelength uncertainties
        )

        super().__init__(**kwargs)

    def evaluate(self, x):
        """
        G25 M33Avg function

        Parameters
        ----------
        x: float
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
        C1 = -2.79
        C2 = 1.46
        B3 = 3.00
        C4 = 0.43
        xo = 4.661
        gamma = 1.41
        C3 = B3 * (gamma**2)

        optnir_axav_x = 1.0 / np.array([1.537, 1.153, 0.805, 0.475, 0.335])
        optnir_axav_y = [0.2295, 0.4266, 0.6394, 1.1237, 1.4024]

        # return A(x)/A(V)
        return _curve_F99_method(
            x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
        )
