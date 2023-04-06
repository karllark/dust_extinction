import warnings

import numpy as np
from scipy.special import comb
import astropy.units as u

__all__ = ["_get_x_in_wavenumbers", "_test_valid_x_range", "_smoothstep"]


def _get_x_in_wavenumbers(in_x):
    """
    Convert input x to wavenumber given x has units.
    Otherwise, assume x is in waveneumbers and issue a warning to this effect.

    Parameters
    ----------
    in_x : astropy.quantity or simple floats
        x values

    Returns
    -------
    x : floats
        input x values in wavenumbers w/o units
    """
    # handles the case where x is a scaler
    in_x = np.atleast_1d(in_x)

    # check if in_x is an astropy quantity, if not issue a warning
    if not isinstance(in_x, u.Quantity):
        warnings.warn(
            "x has no units, assuming x units are inverse microns", UserWarning
        )

    # convert to wavenumbers (1/micron) if x input in units
    # otherwise, assume x in appropriate wavenumber units
    with u.add_enabled_equivalencies(u.spectral()):
        x_quant = u.Quantity(in_x, 1.0 / u.micron, dtype=np.float64)

    # strip the quantity to avoid needing to add units to all the
    #    polynomical coefficients
    return x_quant.value


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
    deltacheck = 1e-6  # delta to allow for small numerical issues
    if np.logical_or(np.any(x <= (x_range[0] - deltacheck)), np.any(x >= (x_range[1] + deltacheck))):
        raise ValueError(
            "Input x outside of range defined for "
            + outname
            + " ["
            + str(x_range[0])
            + " <= x <= "
            + str(x_range[1])
            + ", x has units 1/micron]"
        )


def _smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result
