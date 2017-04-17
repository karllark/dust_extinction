import numpy as np
import pytest

import astropy.units as u
from astropy.modeling import InputParameterError

from ..dust_extinction_averages import G03_SMCBar

x_bad = [-1.0, 0.1, 10.1, 100.]
@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    tmodel = G03_SMCBar()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid)
    assert exc.value.args[0] == 'Input x outside of range defined for G03' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_wavenumber", x_bad/u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    tmodel = G03_SMCBar()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_wavenumber)
    assert exc.value.args[0] == 'Input x outside of range defined for G03' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_micron", u.micron/x_bad)
def test_invalid_micron(x_invalid_micron):
    tmodel = G03_SMCBar()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_micron)
    assert exc.value.args[0] == 'Input x outside of range defined for G03' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'
    
@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom*1e4/x_bad)
def test_invalid_micron(x_invalid_angstrom):
    tmodel = G03_SMCBar()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_angstrom)
    assert exc.value.args[0] == 'Input x outside of range defined for G03' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

def get_axav_cor_vals(index):
    # testing wavenumbers
    x_smc = np.array([0.455, 0.606, 0.800,
                      1.235, 1.538,
                      1.818, 2.273, 2.703,
                      3.375, 3.625, 3.875,
                      4.125, 4.375, 4.625, 4.875,
                      5.125, 5.375, 5.625, 5.875, 
                      6.125, 6.375, 6.625, 6.875, 
                      7.125, 7.375, 7.625, 7.875, 
                      8.125, 8.375, 8.625])
    # last UV point removed as it is poorly fit by FM90 function
    x_lmc = np.array([0.455, 0.606, 0.800,
                      1.818, 2.273, 2.703,
                      3.375, 3.625, 3.875,
                      4.125, 4.375, 4.625, 4.875,
                      5.125, 5.375, 5.625, 5.875, 
                      6.125, 6.375, 6.625, 6.875, 
                      7.125, 7.375, 7.625, 7.875, 
                      8.125])
                  
    # correct values
    if index == 1:
        x = x_smc
        cor_vals = np.array([0.110, 0.169, 0.250,
                             0.567, 0.801,
                             1.000, 1.374, 1.672,
                             2.000, 2.220, 2.428,
                             2.661, 2.947, 3.161, 3.293,
                             3.489, 3.637, 3.866, 4.013,
                             4.243, 4.472, 4.776, 5.000,
                             5.272, 5.575, 5.795, 6.074,
                             6.297, 6.436, 6.992])
    elif index == 2: 
        x = x_lmc
        cor_vals = np.array([0.100, 0.186, 0.257,
                             1.000, 1.293, 1.518,
                             1.786, 1.969, 2.149,
                             2.391, 2.771, 2.967, 2.846,
                             2.646, 2.565, 2.566, 2.598,
                             2.607, 2.668, 2.787, 2.874,
                             2.983, 3.118, 3.231, 3.374,
                             3.366])
    elif index == 3:
        x = x_lmc
        cor_vals = np.array([0.101, 0.150, 0.299,
                             1.000, 1.349, 1.665,
                             1.899, 2.067, 2.249,
                             2.447, 2.777, 2.922, 2.921,
                             2.812, 2.805, 2.863, 2.932,
                             3.060, 3.110, 3.299, 3.408,
                             3.515, 3.670, 3.862, 3.937,
                             4.055])
    else:
        cor_vals = np.array([0])

    # add units
    x = x/u.micron

    return (x, cor_vals)
        
def test_extinction_G03_values():
    # get the correct values
    x, cor_vals = get_axav_cor_vals(1)
    
    # initialize extinction model    
    tmodel = G03_SMCBar()

    # test
    #  not to numerical precision as we are using the FM90 fits
    #  and spline functions and the correct values are the data 
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=6e-02)

test_vals = zip([0.455, 0.606, 0.800,
                 1.235, 1.538,
                 1.818, 2.273, 2.703,
                 3.375, 3.625, 3.875,
                 4.125, 4.375, 4.625, 4.875,
                 5.125, 5.375, 5.625, 5.875, 
                 6.125, 6.375, 6.625, 6.875, 
                 7.125, 7.375, 7.625, 7.875, 
                 8.125, 8.375, 8.625],
                [0.110, 0.169, 0.250,
                 0.567, 0.801,
                 1.000, 1.374, 1.672,
                 2.000, 2.220, 2.428,
                 2.661, 2.947, 3.161, 3.293,
                 3.489, 3.637, 3.866, 4.013,
                 4.243, 4.472, 4.776, 5.000,
                 5.272, 5.575, 5.795, 6.074,
                 6.297, 6.436, 6.992])
@pytest.mark.parametrize("test_vals", test_vals)
def test_extinction_G03_single_values(test_vals):
    x, cor_val = test_vals
    
    # initialize extinction model    
    tmodel = G03_SMCBar()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val, rtol=6e-02)
    
