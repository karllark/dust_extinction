
from __future__ import (absolute_import, print_function, division)

# STDLIB
import warnings

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.modeling import (Model, Fittable1DModel,
                              Parameter, InputParameterError)

__all__ = ['BaseExtModel','BaseExtRvModel', 'BaseExtAve',
           'CCM89', 'FM90', 'P92', 'F99',
           'G03_SMCBar', 'G03_LMCAvg', 'G03_LMC2',
           'G16']

x_range_CCM89 = [0.3,10.0]
x_range_FM90 = [1.0/0.32,1.0/0.0912]
x_range_P92 = [1.0/1e3,1.0/1e-3]
x_range_F99 = [0.3,10.0]
x_range_G03 = [0.3,10.0]
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
        raise ValueError('Input x outside of range defined for ' + outname \
                         + ' ['
                         + str(x_range[0])
                         +  ' <= x <= '
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
    x_splineval_uv = 10000.0/np.array([2700.0,2600.0])

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
        y_splineval_optir   = optnir_axav_y

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
        return np.power(10.0,-0.4*axav*Av)

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
        return np.power(10.0,-0.4*axav*Av)

class BaseExtRvModel(BaseExtModel):
    """
    Base Extinction R(V)-dependent Model.  Do not use.
    """

    Rv = Parameter(description="R(V) = A(V)/E(B-V) = " \
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

    Rv_range = [2.0,6.0]
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
        ir_indxs = np.where(np.logical_and(0.3 <= x,x < 1.1))
        opt_indxs = np.where(np.logical_and(1.1 <= x,x < 3.3))
        nuv_indxs = np.where(np.logical_and(3.3 <= x,x <= 8.0))
        fnuv_indxs = np.where(np.logical_and(5.9 <= x,x <= 8))
        fuv_indxs = np.where(np.logical_and(8 < x,x <= 10))

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
        a[nuv_indxs] = 1.752-.316*x[nuv_indxs] \
                       - 0.104/((x[nuv_indxs] - 4.67)**2 + .341)
        b[nuv_indxs] = -3.09 + \
                       1.825*x[nuv_indxs] \
                       + 1.206/((x[nuv_indxs] - 4.62)**2 + .263)

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
        d_C1 = np.full((len(x)),1.)
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

    #fit_deriv = None

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
                           default=0.08, bounds=(0.07,0.09))
    FUV_b = Parameter(description="FUV term: b coefficient",
                        default=4.0)
    FUV_n = Parameter(description="FUV term: n coefficient",
                        default=6.5)

    NUV_amp = Parameter(description="NUV term: amplitude",
                        default=0.045*AbAv, min=0.0)
    NUV_lambda = Parameter(description="NUV term: center wavelength",
                           default=0.22, bounds=(0.20,0.24))
    NUV_b = Parameter(description="NUV term: b coefficient",
                        default=-1.95)
    NUV_n = Parameter(description="NUV term: n coefficient",
                        default=2.0, fixed=True)

    SIL1_amp = Parameter(description="SIL1 term: amplitude",
                         default=0.002*AbAv, min=0.0)
    SIL1_lambda = Parameter(description="SIL1 term: center wavelength",
                            default=9.7, bounds=(7.0,13.0))
    SIL1_b = Parameter(description="SIL1 term: b coefficient",
                       default=-1.95)
    SIL1_n = Parameter(description="SIL1 term: n coefficient",
                       default=2.0, fixed=True)

    SIL2_amp = Parameter(description="SIL2 term: amplitude",
                         default=0.002*AbAv, min=0.0)
    SIL2_lambda = Parameter(description="SIL2 term: center wavelength",
                            default=18.0, bounds=(15.0,21.0))
    SIL2_b = Parameter(description="SIL2 term: b coefficient",
                        default=-1.80)
    SIL2_n = Parameter(description="SIL2 term: n coefficient",
                        default=2.0, fixed=True)

    FIR_amp = Parameter(description="FIR term: amplitude",
                        default=0.012*AbAv, min=0.0)
    FIR_lambda = Parameter(description="FIR term: center wavelength",
                           default=25.0, bounds=(20.0,30.0))
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

        return amplitude/(np.power(l_norm,n) + np.power(l_norm,-1*n) + b)

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
            + self._p92_single_term(lam, SIL1_amp, SIL1_lambda, SIL1_b, SIL1_n)
            + self._p92_single_term(lam, SIL2_amp, SIL2_lambda, SIL2_b, SIL2_n)
            + self._p92_single_term(lam, FIR_amp, FIR_lambda, FIR_b, FIR_n))

        # return A(x)/A(V)
        return axav

    # use numerical derivaties (need to add analytic)
    fit_deriv = None

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

    Updated for the C1 vs C2 correlation in
       Fitzpatrick & Massa (2007, ApJ, 663, 320)

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
        x = np.arange(text_model.x_range[0], text_model.x_range[1],0.1)/u.micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
           ext_model = F99(Rv=cur_Rv)
           ax.plot(x,ext_model(x),label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    Rv_range = [2.0,6.0]
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
        # updated for FM07 correlation between C1 and C2
        C1 = 2.030 - 3.007*C2

        # spline points
        optnir_axav_x = 10000./np.array([26500.0,12200.0,6000.0,
                                         5470.0,4670.0,4110.0])

        # determine optical/IR values at spline points
        #    Final term has a "-1.208" in Table 4 of F99, but then does
        #    not reproduce Table 3.
        #    Indications are that this is not correct from fm_unred.pro
        #    which is based on FMRCURVE.pro distributed by Fitzpatrick.
        #    --> confirmation needed
        #
        #    Also, fm_unred.pro has different coeff and # of terms, possible
        #    update --> check with Fitzpatrick
        opt_axebv_y = np.array([-0.426 + 1.0044*Rv,
                                -0.050 + 1.0016*Rv,
                                0.701 + 1.0016*Rv,
                                1.208 + 1.0032*Rv - 0.00033*(Rv**2)])
        nir_axebv_y = np.array([0.265,0.829])*Rv/3.1
        optnir_axebv_y = np.concatenate([nir_axebv_y,opt_axebv_y])

        # return A(x)/A(V)
        return _curve_F99_method(in_x, Rv, C1, C2, C3, C4, xo, gamma,
                                 optnir_axav_x, optnir_axebv_y/Rv,
                                 self.x_range, 'F99')

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


class G16(BaseExtModel):
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
        x = np.arange(text_model.x_range[0], text_model.x_range[1],0.1)/u.micron

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
        x = np.arange(text_model.x_range[0], text_model.x_range[1],0.1)/u.micron

        fAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for cur_fA in fAs:
           ext_model = G16(RvA=3.1, fA=cur_fA)
           ax.plot(x,ext_model(x),label=r'$f_A = ' + str(cur_fA) + '$')

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')

        ax.legend(loc='best', title=r'$R_A(V) = 3.1$')
        plt.show()
    """

    RvA = Parameter(description="R_A(V) = A(V)/E(B-V) = " \
                   + "total-to-selective extinction of component A",
                   default=3.1)
    fA = Parameter(description="f_A = mixture coefficent of component A",
                   default=1.0)

    RvA_range = [2.0, 6.0]
    fA_range = [0.0, 1.0]
    x_range = x_range_G16

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
