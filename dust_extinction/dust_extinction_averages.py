
from __future__ import (absolute_import, print_function, division)

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.modeling import Model, Parameter, InputParameterError

from dust_extinction.dust_extinction import FM90

__all__ = ['BaseExtAve', 'G03_SMCBar']

x_range_G03 = [0.3,10.0]

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
                      optnir_axav_x, optnir_axav_y):
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
    _test_valid_x_range(x, x_range_G03, 'G03')
    
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

class BaseExtAve(Model):
    """
    Base Extinction Aveage Model.  Do not use.
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

    Example showing the G03 SMCBar average curve

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.dust_extinction_averages import G03_SMCBar

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.3,10.0,0.1)/u.micron
    
        ext_model = G03_SMCBar()
        ax.plot(x,ext_model(x),label='G03 SMCBar')

        ax.set_xlabel('$x$ [$\mu m^{-1}$]')
        ax.set_ylabel('$A(x)/A(V)$')
    
        ax.legend(loc='best')
        plt.show()
    """

    x_range = x_range_G03

    @staticmethod
    def evaluate(in_x):
        """
        G03 function

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
        Rv = 2.74
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
        return _curve_F99_method(in_x, Rv, C1, C2, C3, C4, xo, gamma,
                                 optnir_axav_x, optnir_axav_y)

    
