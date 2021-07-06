import numpy as np
import pytest
import astropy.units as u

from ..parameter_averages import G16
from ..averages import G03_SMCBar


# used to be in test_f99
def get_axav_cor_vals_fA_1():
    # from Fitzpatrick (1999) Table 3
    x = np.array([0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846])

    cor_vals = np.array([0.265, 0.829, 2.688, 3.055, 3.806, 4.315, 6.265, 6.591])
    tolerance = 2e-3

    # convert from A(x)/E(B-V) to A(x)/A(V)
    cor_vals /= 3.1

    # add units
    x = x / u.micron

    return (x, cor_vals, tolerance)


def test_extinction_G16_fA_1_values():
    # get the correct values
    x, cor_vals, tolerance = get_axav_cor_vals_fA_1()

    # initialize extinction model
    tmodel = G16(RvA=3.1, fA=1.0)

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=tolerance)


def test_extinction_G16_fA_0_values():
    # initialize the model
    tmodel = G16(fA=0.0)

    # get the correct values
    gmodel = G03_SMCBar()
    x = gmodel.obsdata_x
    cor_vals = gmodel.obsdata_axav
    tolerance = gmodel.obsdata_tolerance

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=tolerance)


x_vals, axav_vals, tolerance = get_axav_cor_vals_fA_1()
test_vals = zip(x_vals, axav_vals, np.full(len(x_vals), tolerance))


@pytest.mark.parametrize("test_vals", test_vals)
def test_extinction_G16_fA_1_single_values(test_vals):
    x, cor_val, tolerance = test_vals

    # initialize extinction model
    tmodel = G16(RvA=3.1, fA=1.0)

    # test
    np.testing.assert_allclose(tmodel(x), cor_val, rtol=tolerance)
    np.testing.assert_allclose(tmodel.evaluate(x, 3.1, 1.0), cor_val, rtol=tolerance)


def test_extinction_G16_extinguish_values_Ebv():
    # get the correct values
    x, cor_vals, tolerance = get_axav_cor_vals_fA_1()

    # calculate the cor_vals in fractional units
    Rv = 3.1
    Ebv = 1.0
    Av = Ebv * Rv
    cor_vals = np.power(10.0, -0.4 * (cor_vals * Av))

    # initialize extinction model
    tmodel = G16(RvA=Rv)

    # test
    np.testing.assert_allclose(tmodel.extinguish(x, Ebv=Ebv), cor_vals, rtol=tolerance)
