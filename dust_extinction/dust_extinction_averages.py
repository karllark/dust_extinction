
from __future__ import (absolute_import, print_function, division)

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.modeling import Model, Parameter, InputParameterError

from dust_extinction.dust_extinction import FM90

__all__ = ['BaseExtAve', 'G03_SMCBar', 'G03_LMCAvg', 'G03_LMC2']

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

    Example showing the average curve

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
                                 optnir_axav_x, optnir_axav_y)

    
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

        from dust_extinction.dust_extinction_averages import G03_LMCAvg

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.3,10.0,0.1)/u.micron
    
        ext_model = G03_LMCAvg()
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
                                 optnir_axav_x, optnir_axav_y)

    
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

        from dust_extinction.dust_extinction_averages import G03_LMC2

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.3,10.0,0.1)/u.micron
    
        ext_model = G03_LMC2()
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
                                 optnir_axav_x, optnir_axav_y)

    
