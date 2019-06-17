import numpy as np
import pytest

import astropy.units as u

from ..parameter_averages import GCC09
from .helpers import _invalid_x_range


x_bad = [-1.0, 0.1, 12.0, 100.0]


@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    _invalid_x_range(x_invalid, GCC09(), "GCC09")


@pytest.mark.parametrize("x_invalid_wavenumber", x_bad / u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    _invalid_x_range(x_invalid_wavenumber, GCC09(), "GCC09")


@pytest.mark.parametrize("x_invalid_micron", u.micron / x_bad)
def test_invalid_micron(x_invalid_micron):
    _invalid_x_range(x_invalid_micron, GCC09(), "GCC09")


@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom * 1e4 / x_bad)
def test_invalid_angstrom(x_invalid_angstrom):
    _invalid_x_range(x_invalid_angstrom, GCC09(), "GCC09")


def get_axav_cor_vals(Rv):
    # testing wavenumbers
    x = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.6, 4.0, 3.4])

    # add units
    x = x / u.micron

    # correct values
    # from IDL version
    if Rv == 3.1:
        cor_vals = np.array(
            [
                5.23161,
                4.20810,
                3.45123,
                2.92264,
                2.61283,
                2.85130,
                3.19451,
                2.34301,
                1.89256,
            ]
        )
    elif Rv == 2.0:
        cor_vals = np.array(
            [
                10.5150,
                8.07274,
                6.26711,
                5.00591,
                4.24237,
                4.42844,
                4.99482,
                3.42585,
                2.59322,
            ]
        )
    elif Rv == 3.0:
        cor_vals = np.array(
            [
                5.55181,
                4.44232,
                3.62189,
                3.04890,
                2.71159,
                2.94688,
                3.30362,
                2.40863,
                1.93502,
            ]
        )
    elif Rv == 4.0:
        cor_vals = np.array(
            [
                3.07020,
                2.62711,
                2.29927,
                2.07040,
                1.94621,
                2.20610,
                2.45801,
                1.90003,
                1.60592,
            ]
        )
    elif Rv == 5.0:
        cor_vals = np.array(
            [
                1.58123,
                1.53798,
                1.50571,
                1.48330,
                1.48697,
                1.76164,
                1.95065,
                1.59486,
                1.40846,
            ]
        )
    elif Rv == 6.0:
        cor_vals = np.array(
            [
                0.588581,
                0.811898,
                0.976660,
                1.09190,
                1.18082,
                1.46533,
                1.61241,
                1.39142,
                1.27682,
            ]
        )
    else:
        cor_vals = np.array([0.0])

    return (x, cor_vals)


@pytest.mark.parametrize("Rv", [2.0, 3.0, 3.1, 4.0, 5.0, 6.0])
def test_extinction_GCC09_values(Rv):
    # get the correct values
    x, cor_vals = get_axav_cor_vals(Rv)

    # initialize extinction model
    tmodel = GCC09(Rv=Rv)

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=1e-5)


x_vals, axav_vals = get_axav_cor_vals(3.1)
test_vals = zip(x_vals, axav_vals)


@pytest.mark.parametrize("test_vals", test_vals)
def test_extinction_GCC09_single_values(test_vals):
    x, cor_val = test_vals

    # initialize extinction model
    tmodel = GCC09()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val, rtol=1e-5)
    np.testing.assert_allclose(tmodel.evaluate(x, 3.1), cor_val, rtol=1e-5)
