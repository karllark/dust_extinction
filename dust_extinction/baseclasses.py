from __future__ import (absolute_import, print_function, division)

import numpy as np

from astropy.modeling import (Model, Parameter, InputParameterError)

__all__ = ['BaseExtModel', 'BaseExtAveModel',
           'BaseExtRvModel', 'BaseExtRvAfAModel']


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


class BaseExtAveModel(Model):
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
