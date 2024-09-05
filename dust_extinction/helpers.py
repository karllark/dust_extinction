import warnings

import numpy as np
from scipy.special import comb

from .warnings import SpectralUnitsWarning

__all__ = ["_warn_no_units", "_test_valid_x_range", "_smoothstep"]


def _warn_no_units():
    warnings.warn(
        "x has no units, assuming x units are inverse microns", SpectralUnitsWarning
    )


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
    if np.logical_or(
        np.any(x <= (x_range[0] - deltacheck)), np.any(x >= (x_range[1] + deltacheck))
    ):
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
