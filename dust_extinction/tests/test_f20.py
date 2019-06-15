import numpy as np
import pytest

import astropy.units as u

from ..parameter_averages import F20
from .helpers import _invalid_x_range


x_bad = [-1.0, 0.1, 12.0, 100.0]


@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    _invalid_x_range(x_invalid, F20(), "F20")


@pytest.mark.parametrize("x_invalid_wavenumber", x_bad / u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    _invalid_x_range(x_invalid_wavenumber, F20(), "F20")


@pytest.mark.parametrize("x_invalid_micron", u.micron / x_bad)
def test_invalid_micron(x_invalid_micron):
    _invalid_x_range(x_invalid_micron, F20(), "F20")


@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom * 1e4 / x_bad)
def test_invalid_angstrom(x_invalid_angstrom):
    _invalid_x_range(x_invalid_angstrom, F20(), "F20")


def get_axav_cor_vals():
    # use x values from Fitzpatrick et al. (2000) Table 3
    x = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    cor_vals = np.array(
        [-1.757, -0.629, 0.438, 2.090, 4.139, 5.704, 4.904, 5.684, 7.150]
    )
    tolerance = 2e-3

    # convert from E(x-V)/E(B-V) to A(x)/A(V)
    cor_vals = (cor_vals + 3.1) / 3.1

    # add units
    x = x / u.micron

    return (x, cor_vals, tolerance)


def test_extinction_F20_values():
    # get the correct values
    x, cor_vals, tolerance = get_axav_cor_vals()

    # initialize extinction model
    tmodel = F20()

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=tolerance)


x_vals, axav_vals, tolerance = get_axav_cor_vals()
test_vals = zip(x_vals, axav_vals, np.full(len(x_vals), tolerance))


@pytest.mark.parametrize("xtest_vals", test_vals)
def test_extinction_F20_single_values(xtest_vals):
    x, cor_val, tolerance = xtest_vals

    # initialize extinction model
    tmodel = F20()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val, rtol=tolerance)
    np.testing.assert_allclose(tmodel.evaluate(x, 3.1), cor_val, rtol=tolerance)
