import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter

from .baseclasses import BaseExtModel

__all__ = ["FM90", "FM90_B3", "P92", "G21"]

x_range_FM90 = [1.0 / 0.35, 1.0 / 0.09]
x_range_P92 = [1.0 / 1e3, 1.0 / 1e-3]


def _curve_F99_method(
    x,
    Rv,
    C1,
    C2,
    bump_param,
    C4,
    xo,
    gamma,
    optnir_axav_x,
    optnir_axav_y,
    fm90_version="C3",
):
    """
    Function to return extinction using F99 method

    Parameters
    ----------
    x: float
        expects either x in units of wavelengths or frequency
        or assumes wavelengths in wavenumbers [1/micron]

        internally wavenumbers are used

    Rv: float
       ratio of total to selective extinction = A(V)/E(B-V)

    C1: float
        y-intercept of linear term: FM90 parameter

    C2: float
        slope of liner term: FM90 parameter

    bump_param: float
        C3 for fm90_version = "C3"
        B3 for fm90_version = "B3"

    C4: float
        amplitude of FUV rise: FM90 parameter

    xo: float
        centroid of "2175 A" bump: FM90 parameter

    gamma: float
        width of "2175 A" bump: FM90 parameter

    optnir_axav_x: float array
        vector of x values for optical/NIR A(x)/A(V) curve

    optnir_axav_y: float array
        vector of y values for optical/NIR A(x)/A(V) curve

    fm90_version: str
        "C3" for standard FM90, "B3" for FM90_B3 for true bump amplitude version

    Returns
    -------
    axav: np array (float)
        A(x)/A(V) extinction curve [mag]

    Raises
    ------
    ValueError
        Input x values outside of defined range
    """
    # initialize extinction curve storage
    axav = np.zeros(x.shape)

    # x value above which FM90 parametrization used
    x_cutval_uv = 10000.0 / 2700.0

    # required UV points for spline interpolation
    x_splineval_uv = 10000.0 / np.array([2700.0, 2600.0])

    # UV points in input x
    (indxs_uv,) = np.where(x >= x_cutval_uv)

    # add in required spline points, otherwise just spline points
    if len(indxs_uv) > 0:
        xuv = np.concatenate([x_splineval_uv, x[indxs_uv]])
    else:
        xuv = x_splineval_uv

    # FM90 model and values
    if fm90_version == "B3":
        fm90_model = FM90_B3(C1=C1, C2=C2, B3=bump_param, C4=C4, xo=xo, gamma=gamma)
    else:
        fm90_model = FM90(C1=C1, C2=C2, C3=bump_param, C4=C4, xo=xo, gamma=gamma)
    # evaluate model and get results in A(x)/A(V)
    axav_fm90 = fm90_model(xuv / u.micron) / Rv + 1.0

    # save spline points
    y_splineval_uv = axav_fm90[0:2]

    # ingore the spline points
    if len(indxs_uv) > 0:
        axav[indxs_uv] = axav_fm90[2:]

    # **Optical Portion**
    #   using cubic spline anchored in UV, optical, and IR

    # optical/NIR points in input x
    (indxs_opir,) = np.where(x < x_cutval_uv)

    if len(indxs_opir) > 0:
        # spline points
        x_splineval_optir = optnir_axav_x

        # determine optical/IR values at spline points
        y_splineval_optir = optnir_axav_y

        # add in zero extinction at infinite wavelength
        x_splineval_optir = np.insert(x_splineval_optir, 0, 0.0)
        y_splineval_optir = np.insert(y_splineval_optir, 0, 0.0)

        spline_x = np.concatenate([x_splineval_optir, x_splineval_uv])
        spline_y = np.concatenate([y_splineval_optir, y_splineval_uv])
        axav[indxs_opir] = interpolate.CubicSpline(spline_x, spline_y, bc_type='natural')(x[indxs_opir])

    # return A(x)/A(V)
    return axav


def _modified_drude(x, scale, x_o, gamma_o, asym):
    """
    Modified Drude function to have a variable asymmetry.  Drude profiles
    are intrinsically asymmetric with the asymmetry fixed by specific central
    wavelength and width.  This modified Drude introduces an asymmetry
    parameter that allows for variable asymmetry at fixed central wavelength
    and width.

    Parameters
    ----------
    x : float
        input wavelengths

    scale : float
        central amplitude

    x_o : float
        central wavelength

    gamma_o : float
        full-width-half-maximum of profile

    asym : float
        asymmetry where a value of 0 results in a standard Drude profile
    """
    gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
    y = scale * ((gamma / x_o) ** 2) / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)

    return y


class FM90(BaseExtModel, Fittable1DModel):
    r"""
    Fitzpatrick & Massa (1990) 6 parameter ultraviolet shape model

    Parameters
    ----------
    C1: float
       y-intercept of linear term

    C2: float
       slope of liner term

    C3: float
       strength of "2175 A" bump (true amplitude is C3/gamma^2)

    C4: float
       amplitude of FUV rise

    xo: float
       centroid of "2175 A" bump

    gamma: float
       width of "2175 A" bump

    Notes
    -----
    From Fitzpatrick & Massa (1990, ApJS, 72, 163)

    Only applicable at UV wavelengths

    Example showing a FM90 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.shapes import FM90

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(3.8,8.6,0.1)/u.micron

        ext_model = FM90()
        ax.plot(x,ext_model(x),label='total')

        ext_model = FM90(C3=0.0, C4=0.0)
        ax.plot(x,ext_model(x),label='linear term')

        ext_model = FM90(C1=0.0, C2=0.0, C4=0.0)
        ax.plot(x,ext_model(x),label='bump term')

        ext_model = FM90(C1=0.0, C2=0.0, C3=0.0)
        ax.plot(x,ext_model(x),label='FUV rise term')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$E(\lambda - V)/E(B - V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.12, 0.15, 0.2, 0.3])
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

    n_inputs = 1
    n_outputs = 1

    # bounds based on Gordon et al. (2024) results
    C1 = Parameter(
        description="linear term: y-intercept", default=0.10, bounds=(-10.0, 5.0)
    )
    C2 = Parameter(description="linear term: slope", default=0.70, bounds=(-0.1, 5.0))
    C3 = Parameter(description="bump: amplitude", default=3.23, bounds=(-1.0, 6.0))
    C4 = Parameter(description="FUV rise: amplitude", default=0.41, bounds=(-0.5, 1.5))
    xo = Parameter(description="bump: centroid", default=4.59, bounds=(4.5, 4.9))
    gamma = Parameter(description="bump: width", default=0.95, bounds=(0.6, 1.7))

    x_range = x_range_FM90

    @staticmethod
    def evaluate(x, C1, C2, C3, C4, xo, gamma):
        """
        FM90 function

        Parameters
        ----------
        x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        exvebv: np array (float)
            E(x-V)/E(B-V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # linear term
        exvebv = C1 + C2 * x

        # bump term
        x2 = x**2
        exvebv += C3 * (x2 / ((x2 - xo**2) ** 2 + x2 * (gamma**2)))

        # FUV rise term
        fnuv_indxs = np.where(x >= 5.9)
        if len(fnuv_indxs) > 0:
            y = x[fnuv_indxs] - 5.9
            exvebv[fnuv_indxs] += C4 * (0.5392 * (y**2) + 0.05644 * (y**3))

        # return E(x-V)/E(B-V)
        return exvebv

    @staticmethod
    def fit_deriv(x, C1, C2, C3, C4, xo, gamma):
        """
        Derivatives of the FM90 function with respect to the parameters
        """
        # useful quantitites
        x2 = x**2
        xo2 = xo**2
        g2 = gamma**2
        x2mxo2_2 = (x2 - xo2) ** 2
        denom = (x2mxo2_2 - x2 * g2) ** 2

        # derivatives
        d_C1 = np.full((len(x)), 1.0)
        d_C2 = x

        d_C3 = x2 / (x2mxo2_2 + x2 * g2)

        d_xo = (4.0 * C2 * x2 * xo * (x2 - xo2)) / denom

        d_gamma = (2.0 * C2 * (x2**2) * gamma) / denom

        d_C4 = np.zeros((len(x)))
        fuv_indxs = np.where(x >= 5.9)
        if len(fuv_indxs) > 0:
            y = x[fuv_indxs] - 5.9
            d_C4[fuv_indxs] = 0.5392 * (y**2) + 0.05644 * (y**3)

        return [d_C1, d_C2, d_C3, d_C4, d_xo, d_gamma]

    # @property
    # def input_units(self):
    #     if self.xo.unit is None:
    #         return None
    #     return {self.inputs[0]: self.xo.unit}
    #
    # def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
    #     return {'xo': inputs_unit[self.inputs[0]],
    #             'C1': outputs_unit[self.outputs[0]],
    #             'C2': outputs_unit[self.outputs[0]],
    #             'C3': outputs_unit[self.outputs[0]],
    #             'C4': outputs_unit[self.outputs[0]]}


class FM90_B3(BaseExtModel, Fittable1DModel):
    r"""
    Fitzpatrick & Massa (1990) 6 parameter ultraviolet shape model
    Version with bump amplitude B3 = C3/gamma^2

    Parameters
    ----------
    C1: float
       y-intercept of linear term

    C2: float
       slope of liner term

    B3: float
       amplitude of "2175 A" bump

    C4: float
       amplitude of FUV rise

    xo: float
       centroid of "2175 A" bump

    gamma: float
       width of "2175 A" bump

    Notes
    -----
    From Fitzpatrick & Massa (1990, ApJS, 72, 163)

    Only applicable at UV wavelengths

    Example showing a FM90 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.shapes import FM90_B3

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(3.8,8.6,0.1)/u.micron

        ext_model = FM90_B3()
        ax.plot(x,ext_model(x),label='total')

        ext_model = FM90_B3(B3=0.0, C4=0.0)
        ax.plot(x,ext_model(x),label='linear term')

        ext_model = FM90_B3(C1=0.0, C2=0.0, C4=0.0)
        ax.plot(x,ext_model(x),label='bump term')

        ext_model = FM90_B3(C1=0.0, C2=0.0, B3=0.0)
        ax.plot(x,ext_model(x),label='FUV rise term')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$E(\lambda - V)/E(B - V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.12, 0.15, 0.2, 0.3])
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

    n_inputs = 1
    n_outputs = 1

    # bounds based on Gordon et al. (2024) results
    C1 = Parameter(
        description="linear term: y-intercept", default=0.10, bounds=(-10.0, 5.0)
    )
    C2 = Parameter(description="linear term: slope", default=0.70, bounds=(-0.1, 5.0))
    B3 = Parameter(description="bump: amplitude", default=3.23, bounds=(-1.0, 6.0))
    C4 = Parameter(description="FUV rise: amplitude", default=0.41, bounds=(-0.5, 1.5))
    xo = Parameter(description="bump: centroid", default=4.59, bounds=(4.5, 4.9))
    gamma = Parameter(description="bump: width", default=0.95, bounds=(0.6, 1.7))

    x_range = x_range_FM90

    @staticmethod
    def evaluate(x, C1, C2, B3, C4, xo, gamma):
        """
        FM90 function

        Parameters
        ----------
        x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        exvebv: np array (float)
            E(x-V)/E(B-V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # linear term
        exvebv = C1 + C2 * x

        # bump term
        x2 = x**2
        exvebv += (B3 * gamma**2) * (x2 / ((x2 - xo**2) ** 2 + x2 * (gamma**2)))

        # FUV rise term
        fnuv_indxs = np.where(x >= 5.9)
        if len(fnuv_indxs) > 0:
            y = x[fnuv_indxs] - 5.9
            exvebv[fnuv_indxs] += C4 * (0.5392 * (y**2) + 0.05644 * (y**3))

        # return E(x-V)/E(B-V)
        return exvebv


class P92(BaseExtModel, Fittable1DModel):
    r"""
    Pei (1992) 24 parameter shape model

    Parameters
    ----------
    BKG_amp : float
      background term amplitude
    BKG_lambda : float
      background term central wavelength
    BKG_b : float
      background term b coefficient
    BKG_n : float
      background term n coefficient [FIXED at n = 2]

    FUV_amp : float
      far-ultraviolet term amplitude
    FUV_lambda : float
      far-ultraviolet term central wavelength
    FUV_b : float
      far-ultraviolet term b coefficent
    FUV_n : float
      far-ultraviolet term n coefficient

    NUV_amp : float
      near-ultraviolet (2175 A) term amplitude
    NUV_lambda : float
      near-ultraviolet (2175 A) term central wavelength
    NUV_b : float
      near-ultraviolet (2175 A) term b coefficent
    NUV_n : float
      near-ultraviolet (2175 A) term n coefficient [FIXED at n = 2]

    SIL1_amp : float
      1st silicate feature (~10 micron) term amplitude
    SIL1_lambda : float
      1st silicate feature (~10 micron) term central wavelength
    SIL1_b : float
      1st silicate feature (~10 micron) term b coefficent
    SIL1_n : float
      1st silicate feature (~10 micron) term n coefficient [FIXED at n = 2]

    SIL2_amp : float
      2nd silicate feature (~18 micron) term amplitude
    SIL2_lambda : float
      2nd silicate feature (~18 micron) term central wavelength
    SIL2_b : float
      2nd silicate feature (~18 micron) term b coefficient
    SIL2_n : float
      2nd silicate feature (~18 micron) term n coefficient [FIXED at n = 2]

    FIR_amp : float
      far-infrared term amplitude
    FIR_lambda : float
      far-infrared term central wavelength
    FIR_b : float
      far-infrared term b coefficent
    FIR_n : float
      far-infrared term n coefficient [FIXED at n = 2]

    Notes
    -----
    From Pei (1992, ApJ, 395, 130)

    Applicable from the extreme UV to far-IR

    Example showing a P92 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.shapes import P92

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(-3.0, 3.0, num=1000)
        x = (1.0/lam)/u.micron

        ext_model = P92()
        ax.plot(1/x,ext_model(x),label='total')

        ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                        SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG only')

        ext_model = P92(NUV_amp=0.0,
                        SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+FUV only')

        ext_model = P92(FUV_amp=0.,
                        SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+NUV only')

        ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                        SIL2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+FIR+SIL1 only')

        ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                        SIL1_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+FIR+SIL2 only')

        ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                        SIL1_amp=0.0, SIL2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+FIR only')

        # Milky Way observed extinction as tabulated by Pei (1992)
        MW_x = [0.21, 0.29, 0.45, 0.61, 0.80, 1.11, 1.43, 1.82,
                2.27, 2.50, 2.91, 3.65, 4.00, 4.17, 4.35, 4.57, 4.76,
                5.00, 5.26, 5.56, 5.88, 6.25, 6.71, 7.18, 7.60,
                8.00, 8.50, 9.00, 9.50, 10.00]
        MW_x = np.array(MW_x)
        MW_exvebv = [-3.02, -2.91, -2.76, -2.58, -2.23, -1.60, -0.78, 0.00,
                     1.00, 1.30, 1.80, 3.10, 4.19, 4.90, 5.77, 6.57, 6.23,
                     5.52, 4.90, 4.65, 4.60, 4.73, 4.99, 5.36, 5.91,
                     6.55, 7.45, 8.45, 9.80, 11.30]
        MW_exvebv = np.array(MW_exvebv)
        Rv = 3.08
        MW_axav = MW_exvebv/Rv + 1.0
        ax.plot(1./MW_x, MW_axav, 'o', label='MW Observed')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylim(1e-3,10.)

        ax.set_xlabel(r'$\lambda$ [$\mu$m]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    n_inputs = 1
    n_outputs = 1

    # constant for conversion from Ax/Ab to (more standard) Ax/Av
    AbAv = 1.0 / 3.08 + 1.0

    BKG_amp = Parameter(
        description="BKG term: amplitude", default=165.0 * AbAv, min=0.0
    )
    BKG_lambda = Parameter(description="BKG term: center wavelength", default=0.047)
    BKG_b = Parameter(description="BKG term: b coefficient", default=90.0)
    BKG_n = Parameter(description="BKG term: n coefficient", default=2.0, fixed=True)

    FUV_amp = Parameter(description="FUV term: amplitude", default=14.0 * AbAv, min=0.0)
    FUV_lambda = Parameter(
        description="FUV term: center wavelength", default=0.07, bounds=(0.06, 0.08)
    )
    FUV_b = Parameter(description="FUV term: b coefficient", default=4.0)
    FUV_n = Parameter(description="FUV term: n coefficient", default=6.5)

    NUV_amp = Parameter(
        description="NUV term: amplitude", default=0.045 * AbAv, min=0.0
    )
    NUV_lambda = Parameter(
        description="NUV term: center wavelength", default=0.22, bounds=(0.20, 0.24)
    )
    NUV_b = Parameter(description="NUV term: b coefficient", default=-1.95)
    NUV_n = Parameter(description="NUV term: n coefficient", default=2.0, fixed=True)

    SIL1_amp = Parameter(
        description="SIL1 term: amplitude", default=0.002 * AbAv, min=0.0
    )
    SIL1_lambda = Parameter(
        description="SIL1 term: center wavelength", default=9.7, bounds=(7.0, 13.0)
    )
    SIL1_b = Parameter(description="SIL1 term: b coefficient", default=-1.95)
    SIL1_n = Parameter(description="SIL1 term: n coefficient", default=2.0, fixed=True)

    SIL2_amp = Parameter(
        description="SIL2 term: amplitude", default=0.002 * AbAv, min=0.0
    )
    SIL2_lambda = Parameter(
        description="SIL2 term: center wavelength", default=18.0, bounds=(15.0, 21.0)
    )
    SIL2_b = Parameter(description="SIL2 term: b coefficient", default=-1.80)
    SIL2_n = Parameter(description="SIL2 term: n coefficient", default=2.0, fixed=True)

    FIR_amp = Parameter(
        description="FIR term: amplitude", default=0.012 * AbAv, min=0.0
    )
    FIR_lambda = Parameter(
        description="FIR term: center wavelength", default=25.0, bounds=(20.0, 30.0)
    )
    FIR_b = Parameter(description="FIR term: b coefficient", default=0.00)
    FIR_n = Parameter(description="FIR term: n coefficient", default=2.0, fixed=True)

    x_range = x_range_P92

    @staticmethod
    def _p92_single_term(in_lambda, amplitude, cen_wave, b, n):
        r"""
        Function for calculating a single P92 term

        .. math::

           \frac{a}{(\lambda/cen_wave)^n + (cen_wave/\lambda)^n + b}

        when n = 2, this term is equivalent to a Drude profile

        Parameters
        ----------
        in_lambda: vector of floats
           wavelengths in same units as cen_wave

        amplitude: float
           amplitude

        cen_wave: flaot
           central wavelength

        b : float
           b coefficient

        n : float
           n coefficient
        """
        l_norm = in_lambda / cen_wave

        return amplitude / (np.power(l_norm, n) + np.power(l_norm, -1 * n) + b)

    def evaluate(
        self,
        x,
        BKG_amp,
        BKG_lambda,
        BKG_b,
        BKG_n,
        FUV_amp,
        FUV_lambda,
        FUV_b,
        FUV_n,
        NUV_amp,
        NUV_lambda,
        NUV_b,
        NUV_n,
        SIL1_amp,
        SIL1_lambda,
        SIL1_b,
        SIL1_n,
        SIL2_amp,
        SIL2_lambda,
        SIL2_b,
        SIL2_n,
        FIR_amp,
        FIR_lambda,
        FIR_b,
        FIR_n,
    ):
        """
        P92 function

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
        # calculate the terms
        lam = 1.0 / x
        axav = (
            self._p92_single_term(lam, BKG_amp, BKG_lambda, BKG_b, BKG_n)
            + self._p92_single_term(lam, FUV_amp, FUV_lambda, FUV_b, FUV_n)
            + self._p92_single_term(lam, NUV_amp, NUV_lambda, NUV_b, NUV_n)
            + self._p92_single_term(lam, SIL1_amp, SIL1_lambda, SIL1_b, SIL1_n)
            + self._p92_single_term(lam, SIL2_amp, SIL2_lambda, SIL2_b, SIL2_n)
            + self._p92_single_term(lam, FIR_amp, FIR_lambda, FIR_b, FIR_n)
        )

        # return A(x)/A(V)
        return axav

    # use numerical derivaties (need to add analytic)
    fit_deriv = None


class G21(BaseExtModel, Fittable1DModel):
    r"""
    Gordon et al. (2021) powerlaw plus two modified Drude profiles
    (for the 10 & 20 micron silicate features)
    for the 1 to 40 micron A(lambda)/A(V) extinction curve.

    Parameters
    ----------
    scale: float
        amplitude of the powerlaw at 1 micron
    alpha: float
        power of powerlaw
    sil1_amp: float
        central amplitude of the 10 micron silicate feature
    sil1_center: float
        center wavelength of the 10 micron silicate feature
    sil1_fwhm: float
        full width at half maximum of the 10 micron silicate feature
    sil1_asym: float
        asymmetry of the 10 micron silicate feature
    sil2_amp: float
        central amplitude of the 20 micron silicate feature
    sil2_center: float
        center wavelength of the 20 micron silicate feature
    sil2_fwhm: float
        full width at half maximum of the 20 micron silicate feature
    sil2_asym: float
        asymmetry of the 20 micron silicate feature

    Notes
    -----
    From Gordon et al. (2021, ApJ, submitted)

    Only applicable at NIR/MIR wavelengths from 1-40 micron

    Example showing a G21 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.shapes import G21

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(1.01), np.log10(39.9), num=1000)
        x = (1.0/lam)/u.micron

        ext_model = G21()
        ax.plot(1/x,ext_model(x),label='total')

        ext_model = G21(sil1_amp=0.0, sil2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law only')

        ext_model = G21(sil2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law+sil1 only')

        ext_model = G21(sil1_amp=0.0)
        ax.plot(1./x,ext_model(x),label='power-law+sil2 only')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(r'$\lambda$ [$\mu$m]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.set_title('G21')

        ax.legend(loc='best')
        plt.show()

    """

    # inputs = ("x",)
    # outputs = ("axav",)

    scale = Parameter(
        description="powerlaw: amplitude", default=0.37, bounds=(0.0, 1.0)
    )
    alpha = Parameter(description="powerlaw: alpha", default=1.5, bounds=(0.5, 5.0))
    sil1_amp = Parameter(
        description="silicate 10um: amplitude", default=0.067, bounds=(0.001, 0.3)
    )
    sil1_center = Parameter(
        description="silicate 10um: center", default=9.84, bounds=(8.0, 12.0)
    )
    sil1_fwhm = Parameter(
        description="silicate 10um: fwhm", default=2.21, bounds=(1.0, 10.0)
    )
    sil1_asym = Parameter(
        description="silicate 10um: asymmetry", default=-0.25, bounds=(-2.0, 2.0)
    )
    sil2_amp = Parameter(
        description="silicate 20um: amplitude", default=0.027, bounds=(0.001, 0.3)
    )
    sil2_center = Parameter(
        description="silicate 20um: center", default=19.26, bounds=(16.0, 24.0)
    )
    sil2_fwhm = Parameter(
        description="silicate 20um: fwhm", default=17.0, bounds=(5.0, 20.0)
    )
    sil2_asym = Parameter(
        description="silicate 20um: asymmetry", default=-0.27, bounds=(-2.0, 2.0)
    )

    x_range = [1.0 / 40.0, 1.0]

    def evaluate(
        self,
        x,
        scale,
        alpha,
        sil1_amp,
        sil1_center,
        sil1_fwhm,
        sil1_asym,
        sil2_amp,
        sil2_center,
        sil2_fwhm,
        sil2_asym,
    ):
        """
        G21 function

        Parameters
        ----------
        x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        wave = 1 / x

        # powerlaw
        axav = scale * (wave ** (-1.0 * alpha))

        # silicate feature drudes
        axav += _modified_drude(wave, sil1_amp, sil1_center, sil1_fwhm, sil1_asym)
        axav += _modified_drude(wave, sil2_amp, sil2_center, sil2_fwhm, sil2_asym)

        return axav
