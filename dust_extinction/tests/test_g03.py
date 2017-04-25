import numpy as np
import pytest

import astropy.units as u
from astropy.modeling import InputParameterError

from ..dust_extinction_averages import G03_SMCBar, G03_LMCAvg, G03_LMC2

x_bad = [-1.0, 0.1, 10.1, 100.]
models = [G03_SMCBar(), G03_LMCAvg(), G03_LMC2()]

@pytest.mark.parametrize("x_invalid", x_bad)
@pytest.mark.parametrize("tmodel", models)
def test_invalid_wavenumbers(x_invalid, tmodel):
    
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid)
    assert exc.value.args[0] == 'Input x outside of range defined for G03' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_wavenumber", x_bad/u.micron)
@pytest.mark.parametrize("tmodel", models)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber, tmodel):
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_wavenumber)
    assert exc.value.args[0] == 'Input x outside of range defined for G03' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_micron", u.micron/x_bad)
@pytest.mark.parametrize("tmodel", models)
def test_invalid_micron(x_invalid_micron, tmodel):
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_micron)
    assert exc.value.args[0] == 'Input x outside of range defined for G03' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'
    
@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom*1e4/x_bad)
@pytest.mark.parametrize("tmodel", models)
def test_invalid_micron(x_invalid_angstrom, tmodel):
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_angstrom)
    assert exc.value.args[0] == 'Input x outside of range defined for G03' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("tmodel", models)
def test_extinction_G03_values(tmodel):
    # test
    #  not to numerical precision as we are using the FM90 fits
    #  and spline functions and the correct values are the data 
    np.testing.assert_allclose(tmodel(tmodel.obsdata_x),
                               tmodel.obsdata_axav, rtol=6e-02)

@pytest.mark.parametrize("tmodel", models)
def test_extinction_G03_single_values(tmodel):
    # test
    for x, cor_val in zip(tmodel.obsdata_x, tmodel.obsdata_axav):
        np.testing.assert_allclose(tmodel(x), cor_val, rtol=6e-02)
    
