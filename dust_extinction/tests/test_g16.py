import numpy as np
import pytest

from ..parameter_averages import G16
from ..averages import G03_SMCBar
from .test_f99 import get_axav_cor_vals as get_axav_cor_vals_fA_1


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
