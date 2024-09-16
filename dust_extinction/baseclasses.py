import numpy as np
from scipy.interpolate import interp1d

from astropy.modeling import Model, Parameter, InputParameterError
from astropy import units as u

from .helpers import _warn_no_units, _test_valid_x_range

__all__ = ["BaseExtModel", "BaseExtRvModel", "BaseExtRvAfAModel", "BaseExtGrainModel"]


class BaseExtModel(Model):
    """
    Base Extinction Model.  Do not use directly.
    """

    n_inputs = 1
    n_outputs = 1
    input_units = {"x": u.micron**-1}
    return_units = {"y": u.dimensionless_unscaled}
    input_units_equivalencies = {"x": u.spectral()}
    _input_units_strict = True
    _input_units_allow_dimensionless = True

    def _prepare_input_single(self, x):
        """Check input units and bounds for a single input."""

        # Get the value of the input in the internal units (1 / micron).
        # Because we set the model's input_units_strict and
        # input_units_allow_dimensionless to True, by this point one of the
        # following must hold:
        #   - The input is in units of 1 / micron.
        #   - The input has units of None.
        #   - The input has units of dimensionless_unscaled.
        #   - The input is simple Numpy array and not a Quantity.
        # In the last three cases, we raise a warning that we are assuming
        # that the units are 1 /micron.
        if not isinstance(x, u.Quantity):
            _warn_no_units()
        elif x.unit is None or x.unit is u.dimensionless_unscaled:
            x = x.value
            _warn_no_units()
        else:
            assert x.unit == self.input_units["x"]
            x = x.value

        _test_valid_x_range(x, self.x_range, self.__class__.__name__)
        return x

    def prepare_inputs(self, *args, **kwargs):
        xs, *rest = super().prepare_inputs(*args, **kwargs)
        return [self._prepare_input_single(x) for x in xs], *rest

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
            raise InputParameterError("neither Av or Ebv passed, one required")

        # if Av is not set and Ebv set, convert to Av
        if Av is None:
            Av = self.Rv * Ebv

        # return fractional extinction
        return np.power(10.0, -0.4 * axav * Av)


class BaseExtRvModel(BaseExtModel):
    """
    Base Extinction R(V)-dependent Model.  Do not use directly.
    """

    Rv = Parameter(
        description="R(V) = A(V)/E(B-V) = " + "total-to-selective extinction",
        default=3.1,
    )

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
            raise InputParameterError(
                "parameter Rv must be between "
                + str(self.Rv_range[0])
                + " and "
                + str(self.Rv_range[1])
            )


class BaseExtRvAfAModel(BaseExtModel):
    """
    Base Extinction R(V)_A, f_A -dependent Model.  Do not use directly.
    """

    RvA = Parameter(
        description="R_A(V) = A(V)/E(B-V) = "
        + "total-to-selective extinction of component A",
        default=3.1,
    )
    fA = Parameter(description="f_A = mixture coefficient of component A", default=1.0)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # set Rv so that extinguishing by Ebv works
        # equation 11 in Gordon et al. (2016, ApJ, 826, 104)
        self.Rv = 1.0 / (self.fA / self.RvA + (1 - self.fA) / 2.74)

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
            raise InputParameterError(
                "parameter RvA must be between "
                + str(self.RvA_range[0])
                + " and "
                + str(self.RvA_range[1])
            )

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
            raise InputParameterError(
                "parameter fA must be between "
                + str(self.fA_range[0])
                + " and "
                + str(self.fA_range[1])
            )


class BaseExtGrainModel(BaseExtModel):
    r"""
    Base for Grain Models

    Parameters
    ----------
    None

    Raises
    ------
    None
    """

    def evaluate(self, x):
        """
        Generic dust grain model function

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
        # define the function allowing for spline interpolation
        #   fill value needed to handle numerical issues at the edges
        #   the x values has already been checked to be in range
        f = interp1d(self.data_x, self.data_axav, fill_value="extrapolate")

        return f(x)
