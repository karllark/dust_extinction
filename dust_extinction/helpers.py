from __future__ import (absolute_import, print_function, division)

import numpy as np

__all__ = ['_test_valid_x_range']


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
