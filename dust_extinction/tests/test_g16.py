import numpy as np
import pytest

import astropy.units as u
from astropy.modeling import InputParameterError

from ..parameter_averages import G16
from ..averages import G03_SMCBar
from .test_f99 import get_axav_cor_vals as get_axav_cor_vals_fA_1
from .helpers import _invalid_x_range


@pytest.mark.parametrize("RvA_invalid", [-1.0, 0.0, 1.9, 6.1, 10.])
def test_invalid_RvA_input(RvA_invalid):
    with pytest.raises(InputParameterError) as exc:
        tmodel = G16(RvA=RvA_invalid)
    assert exc.value.args[0] == 'parameter RvA must be between 2.0 and 6.0'


@pytest.mark.parametrize("fA_invalid", [-1.0, -0.1, 1.1, 10.0])
def test_invalid_fA_input(fA_invalid):
    with pytest.raises(InputParameterError) as exc:
        tmodel = G16(fA=fA_invalid)
    assert exc.value.args[0] == 'parameter fA must be between 0.0 and 1.0'


x_bad = [-1.0, 0.1, 12.0, 100.]


@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    _invalid_x_range(x_invalid, G16(), 'G16')


@pytest.mark.parametrize("x_invalid_wavenumber", x_bad/u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    _invalid_x_range(x_invalid_wavenumber, G16(), 'G16')


@pytest.mark.parametrize("x_invalid_micron", u.micron/x_bad)
def test_invalid_micron(x_invalid_micron):
    _invalid_x_range(x_invalid_micron, G16(), 'G16')


@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom*1e4/x_bad)
def test_invalid_micron(x_invalid_angstrom):
    _invalid_x_range(x_invalid_angstrom, G16(), 'G16')


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
