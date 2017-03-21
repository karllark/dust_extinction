# routine to compute the CCM89 R(V) dependent extinction curve
# some code adapted from google code extinct.py by Erik Tollerud
#  adaption needed as we are directly interested in the extinction curve
#  not in dereddening a spectrum (forward instead of reverse modeling)

from __future__ import (absolute_import, print_function, division)

# STDLIB
import warnings

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.modeling import Model, Parameter, InputParameterError

__all__ = ['BaseExtRvModel', 'CCM89', 'FM90', 'F99']

x_range_CCM89 = [0.3,10.0]
x_range_FM90 = [1.0/0.32,1.0/0.0912]
x_range_F99 = [1.0/6.0,8.7]

class BaseExtRvModel(Model):
    """
    Base Extinction R(V)-dependent Model.  Do not use.
    """
    inputs = ('x',)
    outputs = ('axav',)
    
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
        if np.logical_or(np.any(x < x_range_CCM89[0]),
                         np.any(x > x_range_CCM89[1])):
            raise ValueError('Input x outside of range defined for CCM89' \
                             + ' ['
                             + str(x_range_CCM89[0])
                             +  ' <= x <= '
                             + str(x_range_CCM89[1])
                             + ', x has units 1/micron]')
        
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

class FM90(Model):
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
        if np.logical_or(np.any(x < x_range_FM90[0]),
                         np.any(x > x_range_FM90[1])):
            raise ValueError('Input x outside of range defined for FM90' \
                             + ' ['
                             + str(x_range_FM90[0])
                             +  ' <= x <= '
                             + str(x_range_FM90[1])
                             + ', x has units 1/micron]')

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

        # generate the curves and plot them
        x = np.arange(0.2,8.7,0.1)/u.micron
    
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

    @staticmethod
    def evaluate(in_x, Rv):
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
        # convert to wavenumbers (1/micron) if x input in units
        # otherwise, assume x in appropriate wavenumber units
        with u.add_enabled_equivalencies(u.spectral()):
            x_quant = u.Quantity(in_x, 1.0/u.micron, dtype=np.float64)       

        # strip the quantity to avoid needing to add units to all the
        #    polynomical coefficients
        x = x_quant.value

        # check that the wavenumbers are within the defined range
        if np.logical_or(np.any(x < x_range_F99[0]),
                         np.any(x > x_range_F99[1])):
            raise ValueError('Input x outside of range defined for F99' \
                             + ' ['
                             + str(x_range_F99[0])
                             +  ' <= x <= '
                             + str(x_range_F99[1])
                             + ', x has units 1/micron]')
        
        # ensure Rv is a single element, not numpy array
        Rv = Rv[0]

        # initialize extinction curve storage
        axebv = np.zeros(len(x))

        # **Ultraviolet** portion
        #   calculated using the FM90 parameterization

        # constant terms
        C3 = 3.23
        C4 = 0.41
        xo = 4.596
        gamma = 0.99

        # terms depending on Rv
        C2 = -0.824 + 4.717/Rv
        # updated for FM07 correlation between C1 and C2
        C1 = 2.030 - 3.007*C2

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
        # evaluate model and get results in A(x)/E(B-V)
        axebv_fm90 = fm90_model(xuv) + Rv

        # save spline points
        y_splineval_uv = axebv_fm90[0:2]

        # ingore the spline points
        if len(indxs_uv) > 0:
            axebv[indxs_uv] = axebv_fm90[2:]
        
        # **Optical Portion**
        #   using cubic spline anchored in UV, optical, and IR

        # optical/NIR points in input x
        indxs_opir, = np.where(x < x_cutval_uv)

        if len(indxs_opir) > 0:
            # spline points
            x_splineval_optir = 10000./np.array([26500.0,12200.0,6000.0,
                                                 5470.0,4670.0,4110.0])
            # add in zero extinction at infinite wavelength
            x_splineval_optir = np.insert(x_splineval_optir, 0, 0.0)

            # determine optical/IR values at spline points
            y_splineval_opt   = np.array([-0.426 + 1.0044*Rv,
                                          -0.050 + 1.0016*Rv,
                                          0.701 + 1.016*Rv,
                                          1.208 + 1.0032*Rv - 0.00033*(Rv**2)])
            y_splineval_ir    = np.array([0.0,0.265,0.829])*Rv/3.1 
            y_splineval_optir = np.concatenate([y_splineval_ir,y_splineval_opt])

            spline_x = np.concatenate([x_splineval_optir, x_splineval_uv])
            spline_y = np.concatenate([y_splineval_optir, y_splineval_uv])
            spline_rep = interpolate.splrep(spline_x, spline_y)
            axebv[indxs_opir] = interpolate.splev(x[indxs_opir],
                                                  spline_rep, der=0)

        # return A(x)/A(V)
        return axebv/Rv

