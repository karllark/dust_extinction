import numpy as np
import pytest

import astropy.units as u

from ..parameter_averages import M14
from .helpers import _invalid_x_range


x_bad = [-1.0, 0.1, 12.0, 100.]


@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    _invalid_x_range(x_invalid, M14(), 'M14')


@pytest.mark.parametrize("x_invalid_wavenumber", x_bad/u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    _invalid_x_range(x_invalid_wavenumber, M14(), 'M14')


@pytest.mark.parametrize("x_invalid_micron", u.micron/x_bad)
def test_invalid_micron(x_invalid_micron):
    _invalid_x_range(x_invalid_micron, M14(), 'M14')


@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom*1e4/x_bad)
def test_invalid_micron(x_invalid_angstrom):
    _invalid_x_range(x_invalid_angstrom, M14(), 'M14')


def get_axav_cor_vals():
    # choose x values from each portion of the curve
    x = np.array([0.5, 2. , 4. , 6. , 9. ])

    # get values of A(x)/A(V) from direct calculation
    # using R5495 = 3.1
    cor_vals = np.array([0.1323, 1.1412, 2.3111, 2.5219, 4.1340])
    tolerance = 2e-3

    # add units
    x = x/u.micron

    return (x, cor_vals, tolerance)


def test_extinction_M14_values():
    # get the correct values
    x, cor_vals, tolerance = get_axav_cor_vals()

    # initialize extinction model
    tmodel = M14()

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=tolerance)


x_vals, axav_vals, tolerance = get_axav_cor_vals()
test_vals = zip(x_vals, axav_vals, np.full(len(x_vals), tolerance))


@pytest.mark.parametrize("test_vals", test_vals)
def test_extinction_M14_single_values(test_vals):
    x, cor_val, tolerance = test_vals

    # initialize extinction model
    tmodel = M14()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val, rtol=tolerance)
