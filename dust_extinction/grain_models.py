import pkg_resources
from scipy.interpolate import interp1d
import numpy as np

from astropy.table import Table
from astropy.modeling import InputParameterError
from astropy.io.fits import getdata

from .helpers import _get_x_in_wavenumbers, _test_valid_x_range
from .baseclasses import BaseExtModel

__all__ = ["DBP90", "WD01", "D03", "ZDA04", "C11", "J13", "HD23"]


class GMBase(BaseExtModel):
    r"""
    Base for Grain Models

    Parameters
    ----------
    None

    Raises
    ------
    None
    """

    def evaluate(self, in_x):
        """
        WD01 function

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
        #   fill value needed to handle numerical issues at the edges
        #   the x values has already been checked to be in range
        f = interp1d(self.data_x, self.data_axav, fill_value="extrapolate")

        return f(x)


class DBP90(GMBase):
    r"""
    Desert et al (1990) Grain Models

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Desert, Boulanger, & Puget (1990, A&A, 237, 215)
    Computed by DustEm and downloaded from the DustEm website.

    Example showing the possible curves

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import DBP90

        fig, ax = plt.subplots()

        ext_model = DBP90()

        lam = np.logspace(np.log10(1.0/ext_model.x_range[1]),
                          np.log10(1.0/ext_model.x_range[0]),
                          num=1000)
        x = (1.0 / lam) / u.micron

        # define the extinction model
        ax.plot(lam,ext_model(x),label=ext_model.__class__.__name__)

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 1e5, 1.0 / 0.0918]

    possnames = {"MWRV31": ("EXT_DBP90.RES.dat", 3.1)}

    def __init__(self, modelname="MWRV31", **kwargs):
        if modelname not in self.possnames.keys():
            raise InputParameterError("modelname not recognized")
        filename = self.possnames[modelname][0]
        self.Rv = self.possnames[modelname][1]

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + filename,
            data_start=1,
            header_start=None,
            format="ascii.basic",
        )

        self.data_x = 1.0 / a["col1"].data

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.data_x - 1.0 / 0.55))

        self.data_axav = a["col8"].data / a["col8"].data[sindxs[0]]

        # accuracy of the data based on calculations
        self.data_tolerance = 1e-6

        super().__init__(**kwargs)


class WD01(GMBase):
    r"""
    Weingartner & Draine (2001) Grain Models

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Weingartner & Draine (2001, ApJ, 548, 296)

    Example showing the possible curves

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import WD01

        fig, ax = plt.subplots()

        tmod = WD01()
        possmodels = tmod.possnames.keys()

        # generate the curves and plot them
        lam = np.logspace(-2.0, 4.0, num=1000)
        x = (1.0 / lam) / u.micron

        for cmodel in possmodels:
            # define the extinction model
            ext_model = WD01(cmodel)
            ax.plot(lam,ext_model(x),label=cmodel)

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1e-4, 1e2]

    possnames = {
        "MWRV31": ("kext_albedo_WD_MW_3.1_60.dat", 3.1),
        "MWRV40": ("kext_albedo_WD_MW_4.0B_40.dat", 4.0),
        "MWRV55": ("kext_albedo_WD_MW_5.5B_30.dat", 5.5),
        "LMCAvg": ("kext_albedo_WD_LMCavg_20.dat", 3.1),
        "LMC2": ("kext_albedo_WD_LMC2_10.dat", 3.1),
        "SMCBar": ("kext_albedo_WD_SMCbar_0.dat", 3.1),
    }

    def __init__(self, modelname="MWRV31", **kwargs):
        if modelname not in self.possnames.keys():
            raise InputParameterError("modelname not recognized")
        filename = self.possnames[modelname][0]
        self.Rv = self.possnames[modelname][1]

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + filename,
            format="ascii.fixed_width",
            header_start=None,
            data_start=41,
            names=("wave", "albedo", "g", "cext"),
            col_starts=(0, 10, 18, 25),
            col_ends=(9, 16, 24, 34),
        )

        self.data_x = 1.0 / a["wave"].data

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.data_x - 1.0 / 0.55))

        self.data_axav = a["cext"].data / a["cext"].data[sindxs[0]]

        # accuracy of the data based on calculations
        self.data_tolerance = 1e-6

        super().__init__(**kwargs)


class D03(GMBase):
    r"""
    Draine (2003) Grain Models

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

    Example showing the possible curves

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import D03

        fig, ax = plt.subplots()

        tmod = D03()
        possmodels = tmod.possnames.keys()

        # generate the curves and plot them
        lam = np.logspace(-4.0, 4.0, num=1000)
        x = (1.0 / lam) / u.micron

        for cmodel in possmodels:
            # define the extinction model
            ext_model = D03(cmodel)
            ax.plot(lam,ext_model(x),label=cmodel)

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1e-4, 1e4]

    possnames = {
        "MWRV31": ("kext_albedo_WD_MW_3.1_60_D03.dat", 3.1),
        "MWRV40": ("kext_albedo_WD_MW_4.0A_40_D03.dat", 4.0),
        "MWRV55": ("kext_albedo_WD_MW_5.5A_30_D03.dat", 5.5),
    }

    def __init__(self, modelname="MWRV31", **kwargs):
        if modelname not in self.possnames.keys():
            raise InputParameterError("modelname not recognized")
        filename = self.possnames[modelname][0]
        self.Rv = self.possnames[modelname][1]

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

        self.data_x = 1.0 / a["wave"].data

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.data_x - 1.0 / 0.55))

        self.data_axav = a["cext"].data / a["cext"].data[sindxs[0]]

        # accuracy of the data based on calculations
        self.data_tolerance = 1e-6

        super().__init__(**kwargs)


class ZDA04(GMBase):
    r"""
    Zubko, Dwek, & Arendt (2004) Grain Models

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

    Example showing the possible curves

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import ZDA04

        fig, ax = plt.subplots()

        tmod = ZDA04()
        possmodels = tmod.possnames.keys()

        # generate the curves and plot them
        lam = np.logspace(-3.0, 4.0, num=1000)
        x = (1.0 / lam) / u.micron

        for cmodel in possmodels:
            # define the extinction model
            ext_model = ZDA04(cmodel)
            ax.plot(lam,ext_model(x),label=cmodel)

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1e-4, 1e3]

    possnames = {"BARE-GR-S": ("zubko2004_bare-gr-s_alam_av.dat", 3.1)}

    def __init__(self, modelname="BARE-GR-S", **kwargs):
        if modelname not in self.possnames.keys():
            raise InputParameterError("modelname not recognized")
        filename = self.possnames[modelname][0]
        self.Rv = self.possnames[modelname][1]

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + filename,
            format="ascii.basic",
        )

        self.data_x = 1.0 / a["lam[um]"].data

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.data_x - 1.0 / 0.55))

        self.data_axav = a["A_lam/A_V"].data / a["A_lam/A_V"].data[sindxs[0]]

        # accuracy of the data based on calculations
        self.data_tolerance = 1e-6

        super().__init__(**kwargs)


class C11(GMBase):
    r"""
    Compiegne et al (2011) Grain Models

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Compiegne et al. (2011, A&A, 525, 103)
    Computed by DustEm and downloaded from the DustEm website

    Example showing the possible curves

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import C11

        fig, ax = plt.subplots()

        ext_model = C11()

        lam = np.logspace(np.log10(1.0/ext_model.x_range[1]),
                          np.log10(1.0/ext_model.x_range[0]),
                          num=1000)
        x = (1.0 / lam) / u.micron

        # define the extinction model
        ax.plot(lam,ext_model(x),label=ext_model.__class__.__name__)

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 1e5, 1.0 / 4e-2]

    possnames = {"MWRV31": ("EXT_C11.RES.dat", 3.1)}

    def __init__(self, modelname="MWRV31", **kwargs):
        if modelname not in self.possnames.keys():
            raise InputParameterError("modelname not recognized")
        filename = self.possnames[modelname][0]
        self.Rv = self.possnames[modelname][1]

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + filename,
            data_start=1,
            header_start=None,
            format="ascii.basic",
        )

        self.data_x = 1.0 / a["col1"].data

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.data_x - 1.0 / 0.55))

        self.data_axav = a["col12"].data / a["col12"].data[sindxs[0]]

        # accuracy of the data based on calculations
        self.data_tolerance = 1e-6

        super().__init__(**kwargs)


class J13(GMBase):
    r"""
    Jones et al (2013) Grain Models

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Jones et al. (2013, A&A, 558, 62) and Kohler et al. (2014, A&A, 565, 9)
    Computed by DustEm and downloaded from the DustEm website.

    Example showing the possible curves

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import J13

        fig, ax = plt.subplots()

        ext_model = J13()

        lam = np.logspace(np.log10(1.0/ext_model.x_range[1]),
                          np.log10(1.0/ext_model.x_range[0]),
                          num=1000)
        x = (1.0 / lam) / u.micron

        # define the extinction model
        ax.plot(lam,ext_model(x),label=ext_model.__class__.__name__)

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 1e5, 1.0 / 4e-2]

    possnames = {"MWRV31": ("EXT_J13.RES.dat", 3.1)}

    def __init__(self, modelname="MWRV31", **kwargs):
        if modelname not in self.possnames.keys():
            raise InputParameterError("modelname not recognized")
        filename = self.possnames[modelname][0]
        self.Rv = self.possnames[modelname][1]

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = Table.read(
            data_path + filename,
            data_start=1,
            header_start=None,
            format="ascii.basic",
        )

        self.data_x = 1.0 / a["col1"].data

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.data_x - 1.0 / 0.55))

        self.data_axav = a["col10"].data / a["col10"].data[sindxs[0]]

        # accuracy of the data based on calculations
        self.data_tolerance = 1e-6

        super().__init__(**kwargs)


class HD23(GMBase):
    r"""
    Hensley & Draine (2023) Grain Model

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    From Hensley & Draine (2023, ApJ, 948, 55).  File from
    https://dataverse.harvard.edu/dataverse/astrodust

    Example showing the possible curves

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.grain_models import HD23

        fig, ax = plt.subplots()

        ext_model = HD23()

        lam = np.logspace(np.log10(1.0/ext_model.x_range[1]),
                          np.log10(1.0/ext_model.x_range[0]),
                          num=1000)
        x = (1.0 / lam) / u.micron

        # define the extinction model
        ax.plot(lam,ext_model(x),label=ext_model.__class__.__name__)

        ax.set_xlabel(r'$\lambda$ [$\mu m$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = [1.0 / 3e4, 1.0 / 0.1]

    possnames = {"MWRV31": ("astrodust+PAH_MW_RV3.1.fits", 3.1)}

    def __init__(self, modelname="MWRV31", **kwargs):
        if modelname not in self.possnames.keys():
            raise InputParameterError("modelname not recognized")
        filename = self.possnames[modelname][0]
        self.Rv = self.possnames[modelname][1]

        # get the tabulated information
        data_path = pkg_resources.resource_filename("dust_extinction", "data/")

        a = getdata(data_path + filename, 2)

        self.data_x = 1.0 / a[:, 0]

        # normalized by wavelength closest to V band
        sindxs = np.argsort(np.absolute(self.data_x - 1.0 / 0.55))

        self.data_axav = a[:, 3] / a[sindxs[0], 3]

        # accuracy of the data based on calculations
        self.data_tolerance = 1e-6

        super().__init__(**kwargs)
