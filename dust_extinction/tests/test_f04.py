import numpy as np
import pytest

import astropy.units as u

from ..parameter_averages import F04


def get_axav_cor_vals():
    # use x values from Fitzpatrick (1999) Table 3
    x = np.array([0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846])

    # keep optical from Fitzpatrick (1999),
    # replce NIR with Fitzpatrick (2004) function for Rv=3.1:
    # (0.63*3.1 - 0.84)*x**1.84
    cor_vals = np.array([0.185, 0.772, 2.688, 3.055, 3.805, 4.315, 6.456, 6.781])
    tolerance = 2e-3

    # convert from A(x)/E(B-V) to A(x)/A(V)
    cor_vals /= 3.1

    # add units
    x = x / u.micron

    return (x, cor_vals, tolerance)


def test_extinction_F04_values():
    # get the correct values
    x, cor_vals, tolerance = get_axav_cor_vals()

    # initialize extinction model
    tmodel = F04()

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=tolerance)


x_vals, axav_vals, tolerance = get_axav_cor_vals()
test_vals = zip(x_vals, axav_vals, np.full(len(x_vals), tolerance))


@pytest.mark.parametrize("xtest_vals", test_vals)
def test_extinction_F04_single_values(xtest_vals):
    x, cor_val, tolerance = xtest_vals

    # initialize extinction model
    tmodel = F04()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val, rtol=tolerance)
    np.testing.assert_allclose(tmodel.evaluate(x, 3.1), cor_val, rtol=tolerance)
