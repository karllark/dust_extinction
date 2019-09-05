import numpy as np
import pytest

import astropy.units as u

from ..parameter_averages import M14


def get_axav_cor_vals():
    # choose x values from each portion of the curve
    x = np.array([0.5, 2.0])

    # get values of A(x)/A(V) from direct calculation
    # using R5495 = 3.1
    cor_vals = np.array([0.1323, 1.1412])
    tolerance = 2e-3

    # add units
    x = x / u.micron

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
    np.testing.assert_allclose(tmodel.evaluate(x, 3.1), cor_val, rtol=tolerance)
