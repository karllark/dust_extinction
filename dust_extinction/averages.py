from __future__ import absolute_import, print_function, division

import pkg_resources

import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table

from .helpers import _get_x_in_wavenumbers, _test_valid_x_range
from .baseclasses import BaseExtAveModel
from .shapes import P92, _curve_F99_method

__all__ = [
    "RL85_MWGC",
    "G03_SMCBar",
    "G03_LMCAvg",
    "G03_LMC2",
    "I05_MWAvg",
    "CT06_MWGC",
    "CT06_MWLoc",
    "GCC09_MWAvg",
    "F11_MWGC",
]


class RL85_MWGC(BaseExtAveModel):
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

    def evaluate(self, in_x):
        """
        RL85 MWGC function

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

        # define the function using simple linear interpolation
        # avoids negative values of alav that happens with cubic splines
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)


class G03_SMCBar(BaseExtAveModel):
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

    def evaluate(self, in_x):
        """
        G03 SMCBar function

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
            in_x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
            self.x_range,
            self.__class__.__name__,
        )


class G03_LMCAvg(BaseExtAveModel):
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

    def evaluate(self, in_x):
        """
        G03 LMCAvg function

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
            in_x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
            self.x_range,
            self.__class__.__name__,
        )


class G03_LMC2(BaseExtAveModel):
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

    def evaluate(self, in_x):
        """
        G03 LMC2 function

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
            in_x,
            self.Rv,
            C1,
            C2,
            C3,
            C4,
            xo,
            gamma,
            optnir_axav_x,
            optnir_axav_y,
            self.x_range,
            self.__class__.__name__,
        )


class I05_MWAvg(BaseExtAveModel):
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

    def evaluate(self, in_x):
        """
        I05 MWAvg function

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

        # define the function allowing for spline interpolation
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)


class CT06_MWGC(BaseExtAveModel):
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
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + "CT06_pixiedust.dat", format="ascii.commented_header"
        )

        self.obsdata_x = 1.0 / a["wave"].data
        # ext is A(lambda)/A(K)
        # A(K)/A(V) = 0.112 (F19, R(V) = 3.1)
        self.obsdata_axav = 0.112 * a["galcen"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 1e-6

        super().__init__(**kwargs)

    def evaluate(self, in_x):
        """
        CT06 MWGC function

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

        # define the function allowing for spline interpolation
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)


class CT06_MWLoc(BaseExtAveModel):
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
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + "CT06_pixiedust.dat", format="ascii.commented_header"
        )

        self.obsdata_x = 1.0 / a["wave"].data
        # ext is A(lambda)/A(K)
        # A(K)/A(V) = 0.112 (F19, R(V) = 3.1)
        self.obsdata_axav = 0.112 * a["local"].data

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 1e-6

        super().__init__(**kwargs)

    def evaluate(self, in_x):
        """
        CG06 MWLoc function

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

        # define the function allowing for spline interpolation
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)


class GCC09_MWAvg(BaseExtAveModel):
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

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [0.3, 1.0 / 0.0912]

    Rv = 3.1

    def __init__(self, **kwargs):

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        # GCC09 sigma clipped average of 75 sightlines
        a = Table.read(data_path + "GCC09_FUSE.dat", format="ascii.commented_header")
        b = Table.read(data_path + "GCC09_IUE.dat", format="ascii.commented_header")
        c = Table.read(data_path + "GCC09_PHOT.dat", format="ascii.commented_header")

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

    def evaluate(self, in_x):
        """
        GCC09_MWAvg function

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
        return p92_fit(in_x)


class F11_MWGC(BaseExtAveModel):
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
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + "fritz11_galcenter.dat", format="ascii.commented_header"
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

    def evaluate(self, in_x):
        """
        F11 MWGC function

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

        # define the function allowing for spline interpolation
        f = interp1d(self.obsdata_x, self.obsdata_axav)

        return f(x)
