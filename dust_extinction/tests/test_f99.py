import numpy as np
import pytest

import astropy.units as u

from ..parameter_averages import F99


def get_axav_cor_vals():
    # from Fitzpatrick (1999) Table 3
    x = np.array([0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846])

    cor_vals = np.array([0.265, 0.829, 2.688, 3.055, 3.806, 4.315, 6.265, 6.591])
    tolerance = 2e-3

    # convert from A(x)/E(B-V) to A(x)/A(V)
    cor_vals /= 3.1

    # add units
    x = x / u.micron

    return (x, cor_vals, tolerance)


def test_extinction_F99_values():
    # get the correct values
    x, cor_vals, tolerance = get_axav_cor_vals()

    # initialize extinction model
    tmodel = F99()

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=tolerance)


x_vals, axav_vals, tolerance = get_axav_cor_vals()
test_vals = zip(x_vals, axav_vals, np.full(len(x_vals), tolerance))


@pytest.mark.parametrize("test_vals", test_vals)
def test_extinction_F99_single_values(test_vals):
    x, cor_val, tolerance = test_vals

    # initialize extinction model
    tmodel = F99()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val, rtol=tolerance)
    np.testing.assert_allclose(tmodel.evaluate(x, 3.1), cor_val, rtol=tolerance)
