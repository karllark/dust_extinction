
from __future__ import (absolute_import, print_function, division)

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.modeling import (Model, Fittable1DModel,
                              Parameter, InputParameterError)

__all__ = ['CCM89', 'FM90', 'P92', 'O94', 'F99', 'F99FM07'
           'FM07', 'G03_SMCBar', 'G03_LMCAvg', 'G03_LMC2',
           'GCC09_MWAvg', 'G16',
           'AxAvToExv']

x_range_CCM89 = [0.3, 10.0]
x_range_FM90 = [1.0/0.32, 1.0/0.0912]
x_range_P92 = [1.0/1e3, 1.0/1e-3]
x_range_O94 = x_range_CCM89
x_range_F99 = [0.3, 10.0]
x_range_F99FM07 = x_range_F99
x_range_G03 = [0.3, 10.0]
x_range_GCC09 = [0.3, 1.0/0.0912]
x_range_G16 = x_range_G03


def _test_valid_x_range(x, x_range, outname):
    """
    Test if any of the x values are outside of the valid range

    Parameters
    ----------
    x : float array
       wavenumbers in inverse microns

    x_range: 2 floats
       allowed min/max of x

    outname: str
       name of curve for error message
    """
    if np.logical_or(np.any(x < x_range[0]),
                     np.any(x > x_range[1])):
        raise ValueError('Input x outside of range defined for ' + outname
                         + ' ['
                         + str(x_range[0])
                         + ' <= x <= '
                         + str(x_range[1])
                         + ', x has units 1/micron]')


def _curve_F99_method(in_x, Rv,
                      C1, C2, C3, C4, xo, gamma,
                      optnir_axav_x, optnir_axav_y,
                      valid_x_range, model_name):
    """
    Function to return extinction using F99 method

    Parameters
    ----------
    in_x: float
        expects either x in units of wavelengths or frequency
        or assumes wavelengths in wavenumbers [1/micron]

        internally wavenumbers are used

    Rv: float
       ratio of total to selective extinction = A(V)/E(B-V)

    C1: float
        y-intercept of linear term: FM90 parameter

    C2: float
        slope of liner term: FM90 parameter

    C3: float
        amplitude of "2175 A" bump: FM90 parameter

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
    _test_valid_x_range(x, valid_x_range, model_name)

    # initialize extinction curve storage
    axav = np.zeros(len(x))

    # x value above which FM90 parametrization used
    x_cutval_uv = 10000.0/2700.0

    # required UV points for spline interpolation
    x_splineval_uv = 10000.0/np.array([2700.0, 2600.0])

    # UV points in input x
    indxs_uv, = np.where(x >= x_cutval_uv)

    # add in required spline points, otherwise just spline points
    if len(indxs_uv) > 0:
        xuv = np.concatenate([x_splineval_uv, x[indxs_uv]])
    else:
        xuv = x_splineval_uv

    # FM90 model and values
    fm90_model = FM90(C1=C1, C2=C2, C3=C3, C4=C4, xo=xo, gamma=gamma)
    # evaluate model and get results in A(x)/A(V)
    axav_fm90 = fm90_model(xuv)/Rv + 1.0

    # save spline points
    y_splineval_uv = axav_fm90[0:2]

    # ingore the spline points
    if len(indxs_uv) > 0:
        axav[indxs_uv] = axav_fm90[2:]

    # **Optical Portion**
    #   using cubic spline anchored in UV, optical, and IR

    # optical/NIR points in input x
    indxs_opir, = np.where(x < x_cutval_uv)

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
        spline_rep = interpolate.splrep(spline_x, spline_y)
        axav[indxs_opir] = interpolate.splev(x[indxs_opir],
                                             spline_rep, der=0)

    # return A(x)/A(V)
    return axav


class BaseExtModel(Model):
    """
    Base Extinction Model.  Do not use.
    """
    inputs = ('x',)
    outputs = ('axav',)

    def extinguish(self, x, Av=None, Ebv=None):
        """
        Calculate the extinction as a fraction

        Parameters
        ----------
        x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Av: float
           A(V) value of dust column
           Av or Ebv must be set

        Ebv: float
           E(B-V) value of dust column
           Av or Ebv must be set

        Returns
        -------
        frac_ext: np array (float)
           fractional extinction as a function of x
        """
        # get the extinction curve
        axav = self(x)

        # check that av or ebv is set
        if (Av is None) and (Ebv is None):
            raise InputParameterError('neither Av or Ebv passed, one required')

        # if Av is not set and Ebv set, convert to Av
        if Av is None:
            Av = self.Rv*Ebv

        # return fractional extinction
        return np.power(10.0, -0.4*axav*Av)


class BaseExtAve(Model):
    """
    Base Extinction Average.  Do not use.
    """
    inputs = ('x',)
    outputs = ('axav',)

    def extinguish(self, x, Av=None, Ebv=None):
        """
        Calculate the extinction as a fraction

        Parameters
        ----------
        x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Av: float
           A(V) value of dust column
           Av or Ebv must be set

        Ebv: float
           E(B-V) value of dust column
           Av or Ebv must be set

        Returns
        -------
        frac_ext: np array (float)
           fractional extinction as a function of x
        """
        # get the extinction curve
        axav = self(x)

        # check that av or ebv is set
        if (Av is None) and (Ebv is None):
            raise InputParameterError('neither Av or Ebv passed, one required')

        # if Av is not set and Ebv set, convert to Av
        if Av is None:
            Av = self.Rv*Ebv

        # return fractional extinction
        return np.power(10.0, -0.4*axav*Av)


class BaseExtRvModel(BaseExtModel):
    """
    Base Extinction R(V)-dependent Model.  Do not use.
    """
    Rv = Parameter(description="R(V) = A(V)/E(B-V) = "
                   + "total-to-selective extinction",
                   default=3.1)

    @Rv.validator
    def Rv(self, value):
        """
        Check that Rv is in the valid range

        Parameters
        ----------
        value: float
            R(V) value to check

        Raises
        ------
        InputParameterError
           Input Rv values outside of defined range
        """
        if not (self.Rv_range[0] <= value <= self.Rv_range[1]):
            raise InputParameterError("parameter Rv must be between "
                                      + str(self.Rv_range[0])
                                      + " and "
                                      + str(self.Rv_range[1]))


class BaseExtRvAfAModel(BaseExtModel):
    """
    Base Extinction R(V)_A, f_A -dependent Model.  Do not use.
    """

    RvA = Parameter(description="R_A(V) = A(V)/E(B-V) = "
                    + "total-to-selective extinction of component A",
                    default=3.1)
    fA = Parameter(description="f_A = mixture coefficent of component A",
                   default=1.0)

    @RvA.validator
    def RvA(self, value):
        """
        Check that RvA is in the valid range

        Parameters
        ----------
        value: float
            RvA value to check

        Raises
        ------
        InputParameterError
           Input R_A(V) values outside of defined range
        """
        if not (self.RvA_range[0] <= value <= self.RvA_range[1]):
            raise InputParameterError("parameter RvA must be between "
                                      + str(self.RvA_range[0])
                                      + " and "
                                      + str(self.RvA_range[1]))

    @fA.validator
    def fA(self, value):
        """
        Check that fA is in the valid range

        Parameters
        ----------
        value: float
            fA value to check

        Raises
        ------
        InputParameterError
           Input fA values outside of defined range
        """
        if not (self.fA_range[0] <= value <= self.fA_range[1]):
            raise InputParameterError("parameter fA must be between "
                                      + str(self.fA_range[0])
                                      + " and "
                                      + str(self.fA_range[1]))


class AxAvToExv(Fittable1DModel):
    """
    Model to convert from A(x)/A(V) to E(x-V)

    Paramters
    ---------
    Av : float
      dust column in A(V) [mag]
    """
    inputs = ('axav',)
    outputs = ('exv',)

    Av = Parameter(description="A(V)",
                   default=1.0, min=0.0)

    @staticmethod
    def evaluate(axav, Av):
        """
        AlAvToElv function

        Paramters
        ---------
        axav : np array (float)
           E(x-V)/E(B-V) values

        Returns
        -------
        exv : np array (float)
           E(x - V)
        """
        return (axav - 1.0)*Av

    @staticmethod
    def fit_deriv(axav, Av):
        """
        Derivatives of the AxAvtoElv function with respect to the parameters
        """
        # derivatives
        d_Av = (axav - 1.0)

        return [d_Av]


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

        from dust_extinction.dust_extinction import CCM89

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


class FM90(Fittable1DModel):
    """
    FM90 extinction model calculation

    Parameters
    ----------
    C1: float
       y-intercept of linear term

    C2: float
       slope of liner term

    C3: float
       amplitude of "2175 A" bump

    C4: float
       amplitude of FUV rise

    xo: float
       centroid of "2175 A" bump

    gamma: float
       width of "2175 A" bump

    Notes
    -----
    FM90 extinction model

    From Fitzpatrick & Massa (1990)

    Only applicable at UV wavelengths

    Example showing a FM90 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction import FM90

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

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$E(\lambda - V)/E(B - V)$')

        ax.legend(loc='best')
        plt.show()
    """
    inputs = ('x',)
    outputs = ('exvebv',)

    C1 = Parameter(description="linear term: y-intercept",
                   default=0.10)
    C2 = Parameter(description="linear term: slope",
                   default=0.70)
    C3 = Parameter(description="bump: amplitude",
                   default=3.23)
    C4 = Parameter(description="FUV rise: amplitude",
                   default=0.41)
    xo = Parameter(description="bump: centroid",
                   default=4.60)
    gamma = Parameter(description="bump: width",
                      default=0.99)

    x_range = x_range_FM90

    @staticmethod
    def evaluate(in_x, C1, C2, C3, C4, xo, gamma):
        """
        FM90 function

        Parameters
        ----------
        in_x: float
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
        # convert to wavenumbers (1/micron) if x input in units
        # otherwise, assume x in appropriate wavenumber units
        with u.add_enabled_equivalencies(u.spectral()):
            x_quant = u.Quantity(in_x, 1.0/u.micron, dtype=np.float64)

        # strip the quantity to avoid needing to add units to all the
        #    polynomical coefficients
        x = x_quant.value

        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_FM90, 'FM90')

        # linear term
        exvebv = C1 + C2*x

        # bump term
        x2 = x**2
        exvebv += C3*(x2/((x2 - xo**2)**2 + x2*(gamma**2)))

        # FUV rise term
        fnuv_indxs = np.where(x >= 5.9)
        if len(fnuv_indxs) > 0:
            y = x[fnuv_indxs] - 5.9
            exvebv[fnuv_indxs] += C4*(0.5392*(y**2) + 0.05644*(y**3))

        # return E(x-V)/E(B-V)
        return exvebv

    @staticmethod
    def fit_deriv(in_x, C1, C2, C3, C4, xo, gamma):
        """
        Derivatives of the FM90 function with respect to the parameters
        """
        x = in_x

        # useful quantitites
        x2 = x**2
        xo2 = xo**2
        g2 = gamma**2
        x2mxo2_2 = (x2 - xo2)**2
        denom = (x2mxo2_2 - x2*g2)**2

        # derivatives
        d_C1 = np.full((len(x)), 1.)
        d_C2 = x

        d_C3 = (x2/(x2mxo2_2 + x2*g2))

        d_xo = (4.*C2*x2*xo*(x2 - xo2))/denom

        d_gamma = (2.*C2*(x2**2)*gamma)/denom

        d_C4 = np.zeros((len(x)))
        fuv_indxs = np.where(x >= 5.9)
        if len(fuv_indxs) > 0:
            y = x[fuv_indxs] - 5.9
            d_C4[fuv_indxs] = (0.5392*(y**2) + 0.05644*(y**3))

        return [d_C1, d_C2, d_C3, d_C4, d_xo, d_gamma]


class P92(Fittable1DModel):
    """
    P92 extinction model calculation

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
    P92 extinction model

    From Pei (1992)

    Applicable from the extreme UV to far-IR

    Example showing a P92 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction import P92

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

        ax.set_xlabel('$\lambda$ [$\mu$m]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """
    inputs = ('x',)
    outputs = ('axav',)

    # constant for conversion from Ax/Ab to (more standard) Ax/Av
    AbAv = 1.0/3.08 + 1.0

    BKG_amp = Parameter(description="BKG term: amplitude",
                        default=165.*AbAv, min=0.0)
    BKG_lambda = Parameter(description="BKG term: center wavelength",
                           default=0.047)
    BKG_b = Parameter(description="BKG term: b coefficient",
                      default=90.)
    BKG_n = Parameter(description="BKG term: n coefficient",
                      default=2.0, fixed=True)

    FUV_amp = Parameter(description="FUV term: amplitude",
                        default=14.*AbAv, min=0.0)
    FUV_lambda = Parameter(description="FUV term: center wavelength",
                           default=0.07, bounds=(0.06, 0.08))
    FUV_b = Parameter(description="FUV term: b coefficient",
                      default=4.0)
    FUV_n = Parameter(description="FUV term: n coefficient",
                      default=6.5)

    NUV_amp = Parameter(description="NUV term: amplitude",
                        default=0.045*AbAv, min=0.0)
    NUV_lambda = Parameter(description="NUV term: center wavelength",
                           default=0.22, bounds=(0.20, 0.24))
    NUV_b = Parameter(description="NUV term: b coefficient",
                      default=-1.95)
    NUV_n = Parameter(description="NUV term: n coefficient",
                      default=2.0, fixed=True)

    SIL1_amp = Parameter(description="SIL1 term: amplitude",
                         default=0.002*AbAv, min=0.0)
    SIL1_lambda = Parameter(description="SIL1 term: center wavelength",
                            default=9.7, bounds=(7.0, 13.0))
    SIL1_b = Parameter(description="SIL1 term: b coefficient",
                       default=-1.95)
    SIL1_n = Parameter(description="SIL1 term: n coefficient",
                       default=2.0, fixed=True)

    SIL2_amp = Parameter(description="SIL2 term: amplitude",
                         default=0.002*AbAv, min=0.0)
    SIL2_lambda = Parameter(description="SIL2 term: center wavelength",
                            default=18.0, bounds=(15.0, 21.0))
    SIL2_b = Parameter(description="SIL2 term: b coefficient",
                       default=-1.80)
    SIL2_n = Parameter(description="SIL2 term: n coefficient",
                       default=2.0, fixed=True)

    FIR_amp = Parameter(description="FIR term: amplitude",
                        default=0.012*AbAv, min=0.0)
    FIR_lambda = Parameter(description="FIR term: center wavelength",
                           default=25.0, bounds=(20.0, 30.0))
    FIR_b = Parameter(description="FIR term: b coefficient",
                      default=0.00)
    FIR_n = Parameter(description="FIR term: n coefficient",
                      default=2.0, fixed=True)

    x_range = x_range_P92

    @staticmethod
    def _p92_single_term(in_lambda, amplitude, cen_wave, b, n):
        """
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
        l_norm = in_lambda/cen_wave

        return amplitude/(np.power(l_norm, n) + np.power(l_norm, -1*n) + b)

    def evaluate(self, in_x,
                 BKG_amp, BKG_lambda, BKG_b, BKG_n,
                 FUV_amp, FUV_lambda, FUV_b, FUV_n,
                 NUV_amp, NUV_lambda, NUV_b, NUV_n,
                 SIL1_amp, SIL1_lambda, SIL1_b, SIL1_n,
                 SIL2_amp, SIL2_lambda, SIL2_b, SIL2_n,
                 FIR_amp, FIR_lambda, FIR_b, FIR_n):
        """
        P92 function

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
        _test_valid_x_range(x, x_range_P92, 'P92')

        # calculate the terms
        lam = 1.0/x
        axav = (self._p92_single_term(lam, BKG_amp, BKG_lambda, BKG_b, BKG_n)
                + self._p92_single_term(lam, FUV_amp, FUV_lambda, FUV_b, FUV_n)
                + self._p92_single_term(lam, NUV_amp, NUV_lambda, NUV_b, NUV_n)
                + self._p92_single_term(lam, SIL1_amp, SIL1_lambda,
                                        SIL1_b, SIL1_n)
                + self._p92_single_term(lam, SIL2_amp, SIL2_lambda,
                                        SIL2_b, SIL2_n)
                + self._p92_single_term(lam, FIR_amp, FIR_lambda,
                                        FIR_b, FIR_n))

        # return A(x)/A(V)
        return axav

    # use numerical derivaties (need to add analytic)
    fit_deriv = None


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

        from dust_extinction.dust_extinction import O94

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

        from dust_extinction.dust_extinction import F99

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

class F99FM07(BaseExtRvModel):
    """
    F99 extinction model calculation

    Updated with the NIR Rv dependence and C1-C2 relationship in
       Fitzpatrick & Massa (2007, ApJ, 663, 320)



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

    Updated with the NIR Rv dependence and C1-C2 relationship in
    Fitzpatrick & Massa (2007, ApJ, 663, 320)

    Example showing F99FM07 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction import F99FM07

        fig, ax = plt.subplots()

        # temp model to get the correct x range
        text_model = F99FM07()

        # generate the curves and plot them
        x = np.arange(text_model.x_range[0],
                      text_model.x_range[1],0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = F99FM07(Rv=cur_Rv)
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
        F99FM07 function

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
        # using relationship from F99, deprecatetd in FM07
        C2 = -0.824 + 4.717/Rv

        # updated for FM07 correlation between C1 and C2
        C1 = 2.09 - 2.84*C2

        # spline points
        optnir_axav_x = 10000./np.array([26500.0, 12200.0, 6000.0,
                                         5470.0, 4670.0, 4110.0])

        # **Keep optical spline points from F99:
        #    Final optical spline point has a leading "-1.208" in Table 4
        #    of F99, but that does not reproduce Table 3.
        #    Additional indication that this is not correct is from
        #    fm_unred.pro
        #    which is based on FMRCURVE.pro distributed by Fitzpatrick.
        #    --> confirmation needed?
        # **Use NIR spline points from F99 with function in FM07
        opt_axebv_y = np.array([-0.426 + 1.0044*Rv,
                                -0.050 + 1.0016*Rv,
                                0.701 + 1.0016*Rv,
                                1.208 + 1.0032*Rv - 0.00033*(Rv**2)])
        # updated NIR curve, note R dependendence
        nir_axebv_y = (0.63*Rv - 0.83)*(optnir_axav_x[0:1])**1.84

        optnir_axebv_y = np.concatenate([nir_axebv_y, opt_axebv_y])



        # return A(x)/A(V)
        return _curve_F99_method(in_x, Rv, C1, C2, C3, C4, xo, gamma,
                                 optnir_axav_x, optnir_axebv_y/Rv,
                                 self.x_range, 'F99FM07')


class G03_SMCBar(BaseExtAve):
    """
    G03 SMCBar Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    SMCBar G03 average extinction curve

    From Gordon et al. (2003, ApJ, 594, 279)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction import G03_SMCBar

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = G03_SMCBar()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='G03 SMCBar')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = x_range_G03

    Rv = 2.74

    obsdata_x = np.array([0.455, 0.606, 0.800,
                          1.235, 1.538,
                          1.818, 2.273, 2.703,
                          3.375, 3.625, 3.875,
                          4.125, 4.375, 4.625, 4.875,
                          5.125, 5.375, 5.625, 5.875,
                          6.125, 6.375, 6.625, 6.875,
                          7.125, 7.375, 7.625, 7.875,
                          8.125, 8.375, 8.625])
    obsdata_axav = np.array([0.110, 0.169, 0.250,
                             0.567, 0.801,
                             1.000, 1.374, 1.672,
                             2.000, 2.220, 2.428,
                             2.661, 2.947, 3.161, 3.293,
                             3.489, 3.637, 3.866, 4.013,
                             4.243, 4.472, 4.776, 5.000,
                             5.272, 5.575, 5.795, 6.074,
                             6.297, 6.436, 6.992])
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

        optnir_axav_x = 1./np.array([2.198, 1.65, 1.25, 0.81, 0.65,
                                     0.55, 0.44, 0.37])
        # values at 2.198 and 1.25 changed to provide smooth interpolation
        # as noted in Gordon et al. (2016, ApJ, 826, 104)
        optnir_axav_y = [0.11, 0.169, 0.25, 0.567, 0.801,
                         1.00, 1.374, 1.672]

        # return A(x)/A(V)
        return _curve_F99_method(in_x, self.Rv, C1, C2, C3, C4, xo, gamma,
                                 optnir_axav_x, optnir_axav_y,
                                 self.x_range, 'G03')


class G03_LMCAvg(BaseExtAve):
    """
    G03 LMCAvg Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    LMCAvg G03 average extinction curve

    From Gordon et al. (2003, ApJ, 594, 279)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction import G03_LMCAvg

        fig, ax = plt.subplots()

        # define the extinction model
        ext_model = G03_LMCAvg()

        # generate the curves and plot them
        x = np.arange(ext_model.x_range[0], ext_model.x_range[1],0.1)/u.micron

        ax.plot(x,ext_model(x),label='G03 LMCAvg')
        ax.plot(ext_model.obsdata_x, ext_model.obsdata_axav, 'ko',
                label='obsdata')

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = x_range_G03

    Rv = 3.41

    obsdata_x = np.array([0.455, 0.606, 0.800,
                          1.818, 2.273, 2.703,
                          3.375, 3.625, 3.875,
                          4.125, 4.375, 4.625, 4.875,
                          5.125, 5.375, 5.625, 5.875,
                          6.125, 6.375, 6.625, 6.875,
                          7.125, 7.375, 7.625, 7.875,
                          8.125])
    obsdata_axav = np.array([0.100, 0.186, 0.257,
                             1.000, 1.293, 1.518,
                             1.786, 1.969, 2.149,
                             2.391, 2.771, 2.967, 2.846,
                             2.646, 2.565, 2.566, 2.598,
                             2.607, 2.668, 2.787, 2.874,
                             2.983, 3.118, 3.231, 3.374,
                             3.366])
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

        optnir_axav_x = 1./np.array([2.198, 1.65, 1.25,
                                     0.55, 0.44, 0.37])
        # value at 2.198 changed to provide smooth interpolation
        # as noted in Gordon et al. (2016, ApJ, 826, 104) for SMCBar
        optnir_axav_y = [0.10, 0.186, 0.257,
                         1.000, 1.293, 1.518]

        # return A(x)/A(V)
        return _curve_F99_method(in_x, self.Rv, C1, C2, C3, C4, xo, gamma,
                                 optnir_axav_x, optnir_axav_y,
                                 self.x_range, 'G03')


class G03_LMC2(BaseExtAve):
    """
    G03 LMC2 Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    LMC2 G03 average extinction curve

    From Gordon et al. (2003, ApJ, 594, 279)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction import G03_LMC2

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

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = x_range_G03

    Rv = 2.76

    obsdata_x = np.array([0.455, 0.606, 0.800,
                          1.818, 2.273, 2.703,
                          3.375, 3.625, 3.875,
                          4.125, 4.375, 4.625, 4.875,
                          5.125, 5.375, 5.625, 5.875,
                          6.125, 6.375, 6.625, 6.875,
                          7.125, 7.375, 7.625, 7.875,
                          8.125])
    obsdata_axav = np.array([0.101, 0.150, 0.299,
                             1.000, 1.349, 1.665,
                             1.899, 2.067, 2.249,
                             2.447, 2.777, 2.922, 2.921,
                             2.812, 2.805, 2.863, 2.932,
                             3.060, 3.110, 3.299, 3.408,
                             3.515, 3.670, 3.862, 3.937,
                             4.055])
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

        optnir_axav_x = 1./np.array([2.198, 1.65, 1.25,
                                     0.55, 0.44, 0.37])
        # value at 1.65 changed to provide smooth interpolation
        # as noted in Gordon et al. (2016, ApJ, 826, 104) for SMCBar
        optnir_axav_y = [0.101, 0.15, 0.299,
                         1.000, 1.349, 1.665]

        # return A(x)/A(V)
        return _curve_F99_method(in_x, self.Rv, C1, C2, C3, C4, xo, gamma,
                                 optnir_axav_x, optnir_axav_y,
                                 self.x_range, 'G03')


class GCC09_MWAvg(BaseExtAve):
    """
    G09 MW Average Extinction Curve

    Parameters
    ----------
    None

    Raises
    ------
    None

    Notes
    -----
    MW G09 average extinction curve

    From Gordon, Cartledge, & Clayton (2009, ApJ, 705, 1320)

    Example showing the average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction import GCC09_MWAvg

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

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    x_range = x_range_GCC09

    Rv = 3.1

    # GCC09 sigma clipped average of 75 sightlines
    # FUSE range
    obsdata_x_fuse = [10.9546, 10.9103, 10.8662, 10.8223,
                      10.7785, 10.7349, 10.6915, 10.6483, 10.6052, 10.5623,
                      10.5196, 10.4771, 10.4347, 10.3925, 10.3087, 10.267,
                      10.2255, 10.1841, 10.1019, 10.0611, 10.0204, 9.97986,
                      9.93951, 9.89931, 9.85929, 9.81942, 9.77971, 9.70078,
                      9.66155, 9.62249, 9.58357, 9.54482, 9.50623, 9.46779,
                      9.4295, 9.39137, 9.3534, 9.31557, 9.27791, 9.24039,
                      9.20302, 9.16581, 9.12875, 9.09183, 9.05507, 9.01845,
                      8.98199, 8.94567, 8.90949, 8.87346, 8.83758, 8.80185,
                      8.76626, 8.73081, 8.6955, 8.66034, 8.62532, 8.59044,
                      8.55571, 8.52111, 8.48665, 8.45234, 8.41816]
    obsdata_x_fuse = np.array(obsdata_x_fuse)
    obsdata_axav_fuse = [5.66618, 6.40194, 5.77457, 5.79106,
                         5.70592, 5.72051, 5.49183, 5.41765, 5.51848, 5.41397,
                         5.31807, 5.32059, 5.31287, 5.08549, 5.40957, 5.10527,
                         4.9687, 4.90159, 4.80515, 4.87271, 4.93841, 4.72152,
                         4.87334, 4.93926, 4.53512, 4.76922, 4.78194, 4.72528,
                         4.56487, 4.48517, 4.46397, 4.43617, 4.54009, 4.42218,
                         4.3535, 4.24038, 4.31097, 4.20365, 3.99197, 4.17342,
                         4.06771, 4.06164, 4.07516, 4.05083, 3.99641, 3.93076,
                         3.94753, 3.90736, 3.8829, 3.851, 3.80554, 3.77639,
                         3.77054, 3.71965, 3.69324, 3.68308, 3.62794, 3.57599,
                         3.62095, 3.51596, 3.48092, 3.45548, 3.45695]
    obsdata_axav_fuse = np.array(obsdata_axav_fuse)
    obsdata_axav_unc_fuse = [0.841267, 0.564267, 0.14915,
                             0.168216, 0.148527, 0.453437, 0.15257, 0.186925,
                             0.144917, 0.542937, 0.144074, 0.154166, 0.139096,
                             0.142622, 0.149361, 0.152435, 0.138154, 0.22776,
                             0.152251, 0.137892, 0.128214, 0.210881, 0.132385,
                             0.532321, 0.234241, 0.130432, 0.141754, 0.126149,
                             0.135772, 0.133049, 0.113491, 0.112876, 0.130662,
                             0.112918, 0.112409, 0.134404, 0.108662, 0.110052,
                             0.127927, 0.105854, 0.101012, 0.101105, 0.103142,
                             0.100781, 0.0992076, 0.100922, 0.0980583, .095052,
                             0.0977723, 0.0999428, 0.0968485, 0.0935959,
                             0.0962254, 0.0934298, 0.0901496, 0.0914562,
                             0.0869283, 0.0827114, 0.0839611, 0.0824688,
                             0.0881894, 0.0836739, 0.0927445]
    obsdata_axav_unc_fuse = np.array(obsdata_axav_unc_fuse)
    # IUE range
    obsdata_x_iue = [8.67238, 8.63766, 8.60307, 8.56862, 8.53431, 8.50013,
                     8.46609, 8.43219, 8.39843, 8.06812, 8.03582, 8.00364,
                     7.97159, 7.93967, 7.90787, 7.87621, 7.84467, 7.81326,
                     7.78197, 7.75081, 7.71977, 7.68886, 7.65807, 7.6274,
                     7.59686, 7.56644, 7.53614, 7.50596, 7.47591, 7.44597,
                     7.41616, 7.38646, 7.35688, 7.32742, 7.29808, 7.26886,
                     7.23975, 7.21076, 7.18189, 7.15313, 7.12448, 7.09595,
                     7.06754, 7.03924, 7.01105, 6.98298, 6.95501, 6.92716,
                     6.89943, 6.8718, 6.84428, 6.81687, 6.78958, 6.76239,
                     6.73531, 6.70834, 6.68148, 6.65472, 6.62807, 6.60153,
                     6.5751, 6.54877, 6.52255, 6.49643, 6.47041, 6.4445,
                     6.4187, 6.393, 6.3674, 6.3419, 6.3165, 6.29121,
                     6.26602, 6.24093, 6.21594, 6.19105, 6.16625, 6.14156,
                     6.11697, 6.09247, 6.06808, 6.04378, 6.01958, 5.99547,
                     5.97147, 5.94755, 5.92374, 5.90002, 5.87639, 5.85286,
                     5.82942, 5.80608, 5.78283, 5.75968, 5.73661, 5.71364,
                     5.69076, 5.66797, 5.64528, 5.62267, 5.60016, 5.57773,
                     5.5554, 5.53315, 5.51099, 5.48893, 5.46695, 5.44505,
                     5.42325, 5.40153, 5.3799, 5.35836, 5.33691, 5.31553,
                     5.29425, 5.27305, 5.25193, 5.2309, 5.20996, 5.18909,
                     5.16832, 5.14762, 5.12701, 5.10648, 5.08603, 5.06566,
                     5.04538, 5.02517, 5.00505, 4.98501, 4.96505, 4.94517,
                     4.92536, 4.90564, 4.886, 4.86643, 4.84695, 4.82754,
                     4.80821, 4.78895, 4.76978, 4.75068, 4.73165, 4.71271,
                     4.69383, 4.67504, 4.65632, 4.63767, 4.6191, 4.60061,
                     4.58218, 4.56383, 4.54556, 4.52736, 4.50923, 4.49117,
                     4.47319, 4.45528, 4.43744, 4.41967, 4.40197, 4.38434,
                     4.36678, 4.3493, 4.33188, 4.31454, 4.29726, 4.28005,
                     4.26291, 4.24584, 4.22884, 4.21191, 4.19504, 4.17824,
                     4.16151, 4.14485, 4.12825, 4.11172, 4.09525, 4.07886,
                     4.06252, 4.04626, 4.03005, 4.01391, 3.99784, 3.98183,
                     3.96589, 3.95001, 3.93419, 3.91844, 3.90275, 3.88712,
                     3.87155, 3.85605, 3.84061, 3.82523, 3.80991, 3.79466,
                     3.77946, 3.76433, 3.74925, 3.73424, 3.71929, 3.70439,
                     3.68956, 3.67479, 3.66007, 3.64541, 3.63082, 3.61628,
                     3.6018, 3.58737, 3.57301, 3.5587, 3.54445, 3.53026,
                     3.51612, 3.50204, 3.48802, 3.47405, 3.46014, 3.44628,
                     3.43248, 3.41874, 3.40505, 3.39141, 3.37783, 3.36431,
                     3.35084, 3.33742, 3.32405, 3.31074, 3.29749, 3.28428,
                     3.27113, 3.25803, 3.24499, 3.23199, 3.21905, 3.20616,
                     3.19332, 3.18053, 3.1678, 3.15511, 3.14248, 3.1299]
    obsdata_x_iue = np.array(obsdata_x_iue)
    obsdata_axav_iue = [3.65928, 3.63298, 3.62795, 3.6208, 3.64341, 3.53765,
                        3.53677, 3.5427, 3.56652, 3.29218, 3.23503, 3.2238,
                        3.22317, 3.22866, 3.20417, 3.15837, 3.16327, 3.1708,
                        3.1334, 3.12145, 3.11992, 3.07658, 3.08156, 3.04455,
                        3.02463, 3.01345, 3.02961, 2.9983, 2.97199, 2.97655,
                        2.95303, 2.92917, 2.9315, 2.90957, 2.92098, 2.89773,
                        2.9216, 2.96812, 2.92843, 2.84994, 2.76261, 2.77898,
                        2.79977, 2.80212, 2.75931, 2.75237, 2.74339, 2.75293,
                        2.75406, 2.73118, 2.71888, 2.7038, 2.69887, 2.69127,
                        2.67574, 2.68104, 2.67549, 2.67603, 2.64334, 2.65192,
                        2.65471, 2.63698, 2.79805, 2.81398, 2.61601, 2.52357,
                        2.56245, 2.57612, 2.57371, 2.57366, 2.58391, 2.56651,
                        2.53891, 2.5498, 2.547, 2.52921, 2.53505, 2.53957,
                        2.49688, 2.50546, 2.5158, 2.53622, 2.51002, 2.51476,
                        2.50015, 2.51407, 2.49824, 2.50799, 2.49206, 2.51322,
                        2.53216, 2.49304, 2.50232, 2.49258, 2.49553, 2.49845,
                        2.48743, 2.49823, 2.51739, 2.51013, 2.51298, 2.5099,
                        2.51117, 2.50904, 2.50831, 2.52974, 2.52566, 2.53674,
                        2.55254, 2.542, 2.53676, 2.54664, 2.54644, 2.55759,
                        2.5741, 2.58273, 2.61147, 2.61443, 2.62011, 2.62936,
                        2.63393, 2.65589, 2.66957, 2.68421, 2.70437, 2.70523,
                        2.70803, 2.7383, 2.76815, 2.78111, 2.77637, 2.78608,
                        2.79124, 2.84364, 2.84704, 2.84201, 2.91489, 2.89018,
                        2.91718, 2.94838, 2.98511, 2.98107, 3.03746, 2.99576,
                        3.05553, 3.07511, 3.07822, 3.06783, 3.09312, 3.05633,
                        3.05563, 3.09865, 3.05498, 3.06246, 3.02764, 2.99527,
                        3.00572, 2.95838, 2.95917, 2.91366, 2.88411, 2.85771,
                        2.82407, 2.77112, 2.73591, 2.72641, 2.68104, 2.62841,
                        2.63436, 2.57767, 2.58473, 2.54006, 2.51824, 2.4764,
                        2.45969, 2.43959, 2.41843, 2.37988, 2.36786, 2.33728,
                        2.31928, 2.29395, 2.27981, 2.28779, 2.22856, 2.23069,
                        2.19755, 2.20688, 2.19122, 2.1896, 2.15048, 2.11892,
                        2.10666, 2.10841, 2.11195, 2.08599, 2.07774, 2.06357,
                        2.04322, 2.04653, 2.05064, 2.02805, 2.01393, 2.0147,
                        1.98255, 1.96987, 1.97065, 1.9445, 1.93309, 1.95049,
                        1.92026, 1.8823, 1.90472, 1.91047, 1.90488, 1.87021,
                        1.88634, 1.87464, 1.88597, 1.84753, 1.86308, 1.8393,
                        1.83982, 1.81921, 1.81612, 1.79369, 1.82415, 1.80237,
                        1.80587, 1.7588, 1.77455, 1.76309, 1.76982, 1.73848,
                        1.74919, 1.73011, 1.7181, 1.70523, 1.69983, 1.70104,
                        1.67963, 1.69661, 1.69288, 1.64255, 1.69986, 1.64438]
    obsdata_axav_iue = np.array(obsdata_axav_iue)
    axav_unc_iue = [0.0872464, 0.0870429, 0.0884721, 0.0856429, 0.085733,
                    0.0870081, 0.0868195, 0.0824011, 0.0831442, 0.0700169,
                    0.0753427, 0.0706412, 0.0708828, 0.0725433, 0.0718813,
                    0.0733914, 0.0746335, 0.070549, 0.0701482, 0.0751515,
                    0.0703968, 0.0704839, 0.0716527, 0.0665725, 0.0645906,
                    0.0646839, 0.0667977, 0.0638441, 0.0644936, 0.0646092,
                    0.0632691, 0.062183, 0.0614076, 0.0609257, 0.062895,
                    0.0621621, 0.0616836, 0.0667685, 0.0637902, 0.0628244,
                    0.06145, 0.0599388, 0.0582835, 0.0588209, 0.0552912,
                    0.0570526, 0.0570427, 0.0551712, 0.0551769, 0.0532115,
                    0.0537545, 0.05434, 0.0542661, 0.0532496, 0.052242,
                    0.0526348, 0.0528077, 0.0543117, 0.0530651, 0.0543711,
                    0.0535841, 0.0527281, 0.0615197, 0.0744345, 0.0604337,
                    0.0540828, 0.0502652, 0.0493793, 0.0493413, 0.0494069,
                    0.0490926, 0.0485171, 0.0476265, 0.0488545, 0.048111,
                    0.0465552, 0.0456084, 0.0465916, 0.0438029, 0.0443429,
                    0.0436129, 0.0431271, 0.0438585, 0.0437505, 0.0432799,
                    0.0428038, 0.0418432, 0.0427744, 0.0426111, 0.0430678,
                    0.043576, 0.0449369, 0.0427624, 0.0420389, 0.0433408,
                    0.0431901, 0.0408719, 0.0410534, 0.0432204, 0.0428762,
                    0.0412307, 0.0434997, 0.0390442, 0.0404766, 0.0409797,
                    0.0407884, 0.0407895, 0.0418381, 0.0441335, 0.0435533,
                    0.0417593, 0.0411644, 0.0394862, 0.0421479, 0.0443296,
                    0.0444373, 0.0450305, 0.0445311, 0.0444786, 0.0454764,
                    0.0437134, 0.0434556, 0.0441117, 0.045317, 0.0426919,
                    0.0407817, 0.0415202, 0.0436795, 0.0460842, 0.0441931,
                    0.0429678, 0.0416575, 0.0405409, 0.0449516, 0.0422263,
                    0.04354, 0.0447088, 0.0427848, 0.0427436, 0.0454757,
                    0.0459811, 0.0452352, 0.0473206, 0.0440146, 0.0460769,
                    0.0476268, 0.0476545, 0.0466497, 0.0461933, 0.0468001,
                    0.0456869, 0.0484227, 0.0460075, 0.0460413, 0.0450113,
                    0.0462236, 0.0468239, 0.0438579, 0.0437953, 0.0440855,
                    0.0427247, 0.0428557, 0.040375, 0.0404052, 0.038604,
                    0.0383919, 0.0376932, 0.0359274, 0.0366497, 0.0347373,
                    0.0364086, 0.0331321, 0.0339372, 0.0342494, 0.0340913,
                    0.0320641, 0.0325347, 0.03207, 0.0303993, 0.0301872,
                    0.0304542, 0.0295561, 0.0285186, 0.0300075, 0.0277648,
                    0.0288373, 0.0272505, 0.0259275, 0.026904, 0.0273099,
                    0.0270734, 0.0255038, 0.0250177, 0.0253357, 0.0243852,
                    0.0235365, 0.0243051, 0.0234035, 0.0243984, 0.0238785,
                    0.0233703, 0.0230637, 0.0237698, 0.0217698, 0.0231451,
                    0.0234694, 0.0221487, 0.0220043, 0.0225651, 0.0218849,
                    0.0214238, 0.0226075, 0.0199234, 0.0210254, 0.0208846,
                    0.0208468, 0.021534, 0.0201173, 0.0219919, 0.019761,
                    0.0205099, 0.0203457, 0.0206186, 0.0210671, 0.0209995,
                    0.0201318, 0.0199858, 0.0191492, 0.0185285, 0.0190833,
                    0.0181813, 0.0176767, 0.0188425, 0.0177012, 0.0170917,
                    0.0177921, 0.0158575, 0.0162902, 0.0180801, 0.0178876,
                    0.0183926, 0.0195394, 0.0186493, 0.0179556, 0.0174061,
                    0.017488]
    obsdata_axav_unc_iue = np.array(axav_unc_iue)
    # Opt/NIR range
    obsdata_x_bands = np.array([2.73224, 2.28311, 0.819672, 0.613497,
                                0.456621])
    obsdata_axav_bands = np.array([1.53296, 1.30791, 0.291042, 0.188455,
                                   0.095588])
    obsdata_axav_unc_bands = np.array([0.0105681, 0.00506663, 0.00407895,
                                       0.00307513, 0.00371036])

    # put them together
    obsdata_x = np.concatenate((obsdata_x_fuse, obsdata_x_iue,
                                obsdata_x_bands))
    obsdata_axav = np.concatenate((obsdata_axav_fuse, obsdata_axav_iue,
                                   obsdata_axav_bands))
    obsdata_axav_unc = np.concatenate((obsdata_axav_unc_fuse,
                                       obsdata_axav_unc_iue,
                                       obsdata_axav_unc_bands))

    # accuracy of the observed data based on published table
    obsdata_tolerance = 5e-1

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
        # convert to wavenumbers (1/micron) if x input in units
        with u.add_enabled_equivalencies(u.spectral()):
            x_quant = u.Quantity(in_x, 1.0/u.micron, dtype=np.float64)

        # strip the quantity to avoid needing to add units to all the
        #    polynomical coefficients
        x = x_quant.value
        # check that the wavenumbers are within the defined range
        _test_valid_x_range(x, x_range_GCC09, 'GCC09')

        # P92 parameters fit to the data using uncs as weights
        p92_fit = P92(BKG_amp=203.805939127, BKG_lambda=0.0508199427208,
                      BKG_b=88.0591826413, BKG_n=2.0,
                      FUV_amp=5.33962141873, FUV_lambda=0.08,
                      FUV_b=-0.777129536415, FUV_n=3.88322376926,
                      NUV_amp=0.0447023090042, NUV_lambda=0.217548391182,
                      NUV_b=-1.95723797612, NUV_n=2.0,
                      SIL1_amp=0.00264935064935, SIL1_lambda=9.7,
                      SIL1_b=-1.95, SIL1_n=2.0,
                      SIL2_amp=0.00264935064935, SIL2_lambda=18.0,
                      SIL2_b=-1.80, SIL2_n=2.0,
                      FIR_amp=0.01589610389, FIR_lambda=25.0,
                      FIR_b=0.0, FIR_n=2.0)

        # return A(x)/A(V)
        return p92_fit(in_x)


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

        from dust_extinction.dust_extinction import G16

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

        from dust_extinction.dust_extinction import G16

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
