import pkg_resources
from scipy.interpolate import interp1d
import numpy as np

from astropy.table import Table

from .helpers import _get_x_in_wavenumbers, _test_valid_x_range
from .baseclasses import BaseExtModel

__all__ = ["D03_MWRV31", "D03_MWRV40", "D03_MWRV55", "ZDA04_MWRV31"]


class D03_Base(BaseExtModel):
    r"""
    Draine (2003) Grain Model Base

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Draine (2003, ARA&A, 41, 241; 2003, ApJ, 598, 1017).
    Using Weingartner & Draine (2001, ApJ, 548, 296) size distributions
    """

    x_range = [1e-4, 1e4]

    def __init__(self, filename, **kwargs):

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + filename,
            format="ascii.fixed_width",
            header_start=None,
            data_start=67,
            names=("wave", "albedo", "g", "cext"),
            col_starts=(0, 12, 19, 26),
            col_ends=(11, 18, 25, 35),
        )

        self.obsdata_x = 1.0 / a["wave"].data

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.obsdata_x - 1.0 / 0.55))

        # ext is A(lambda)/A(K)
        # A(K)/A(V) = 0.112 (F19, R(V) = 3.1)
        self.obsdata_axav = a["cext"].data / a["cext"].data[sindxs[0]]

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 1e-6

        super().__init__(**kwargs)

    def evaluate(self, in_x):
        """
        D03 function

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


class D03_MWRV31(D03_Base):
    r"""
    Draine (2003) MW RV=3.1 Grain Model Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Draine (2003, ARA&A, 41, 241; 2003, ApJ, 598, 1017).
    Using Weingartner & Draine (2001, ApJ, 548, 296) size distributions

    Example showing the curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import D03_MWRV31

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = D03_MWRV31()

        # generate the curves and plot them
        lam = np.logspace(-4.0, 4.0, num=1000)
        x = (1.0 / lam) / u.micron

        ax.plot(1.0/x,ext_model(x),label='D03_MWRV31')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    Rv = 3.1

    def __init__(self, **kwargs):

        super().__init__("kext_albedo_WD_MW_3.1_60_D03.dat", **kwargs)


class D03_MWRV40(D03_Base):
    r"""
    Draine (2003) MW RV=4.0 Grain Model Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Draine (2003, ARA&A, 41, 241; 2003, ApJ, 598, 1017).
    Using Weingartner & Draine (2001, ApJ, 548, 296) size distributions

    Example showing the curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import D03_MWRV40

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = D03_MWRV40()

        # generate the curves and plot them
        lam = np.logspace(-4.0, 4.0, num=1000)
        x = (1.0 / lam) / u.micron

        ax.plot(1.0/x,ext_model(x),label='D03_MWRV40')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    Rv = 4.0

    def __init__(self, **kwargs):

        super().__init__("kext_albedo_WD_MW_4.0A_40_D03.dat", **kwargs)


class D03_MWRV55(D03_Base):
    r"""
    Draine (2003) MW RV=5.5 Grain Model Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Draine (2003, ARA&A, 41, 241; 2003, ApJ, 598, 1017).
    Using Weingartner & Draine (2001, ApJ, 548, 296) size distributions

    Example showing the curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import D03_MWRV55

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = D03_MWRV55()

        # generate the curves and plot them
        lam = np.logspace(-4.0, 4.0, num=1000)
        x = (1.0 / lam) / u.micron

        ax.plot(1.0/x,ext_model(x),label='D03_MWRV55')

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    Rv = 5.5

    def __init__(self, **kwargs):

        super().__init__("kext_albedo_WD_MW_5.5A_30_D03.dat", **kwargs)


class ZDA04_MWRV31(BaseExtModel):
    r"""
    Zubko, Dwek, Arendt Milky Way R(V)=3.1 Grain Model Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    BARE-GR-S model from Zubko, Dwek, & Arendt (2004, ApJS, 152, 211)
    Calculated by K. Misselt using ZDA04 grain size distributions and compositions
    """

    x_range = [1e-4, 1e3]

    Rv = 3.1

    def __init__(self, **kwargs):

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + "zubko2004_bare-gr-s_alam_av.dat",
            format="ascii.basic",
        )

        self.obsdata_x = 1.0 / a["lam[um]"].data

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.obsdata_x - 1.0 / 0.55))

        # ext is A(lambda)/A(K)
        # A(K)/A(V) = 0.112 (F19, R(V) = 3.1)
        self.obsdata_axav = a["A_lam/A_V"].data / a["A_lam/A_V"].data[sindxs[0]]

        # accuracy of the observed data based on published table
        self.obsdata_tolerance = 1e-6

        super().__init__(**kwargs)

    def evaluate(self, in_x):
        """
        ZDA04_MWRV31 function

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
