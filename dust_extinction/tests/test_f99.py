import numpy as np
import pytest

import astropy.units as u
from astropy.modeling import InputParameterError

from ..dust_extinction import F99

x_bad = [-1.0, 0.1, 12.0, 100.]
@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    tmodel = F99()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid)
    assert exc.value.args[0] == 'Input x outside of range defined for F99' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_wavenumber", x_bad/u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    tmodel = F99()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_wavenumber)
    assert exc.value.args[0] == 'Input x outside of range defined for F99' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_micron", u.micron/x_bad)
def test_invalid_micron(x_invalid_micron):
    tmodel = F99()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_micron)
    assert exc.value.args[0] == 'Input x outside of range defined for F99' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom*1e4/x_bad)
def test_invalid_micron(x_invalid_angstrom):
    tmodel = F99()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_angstrom)
    assert exc.value.args[0] == 'Input x outside of range defined for F99' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

def get_axav_cor_vals():
    # testing wavenumbers
    #  from previous version of code
    #x = np.array([0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0])

    # correct values (not quite right)
    #cor_vals = np.array([0.124997,  0.377073,  1.130636,  1.419644,
    #                     1.630377,  1.888546,  2.275900,  3.014577,
    #                     2.762256,  2.475272,  2.711508,  3.197144])

    # from Fitzpatrick (1999) Table 3
    x = np.array([0.377, 0.820, 1.667, 1.828, 2.141, 2.433,
                  3.704, 3.846])

    cor_vals = np.array([0.265, 0.829, 2.688, 3.055, 3.806, 4.315,
                         6.265, 6.591])
    tolerance = 2e-3

    # convert from A(x)/E(B-V) to A(x)/A(V)
    cor_vals /= 3.1

    # add units
    x = x/u.micron

    return (x, cor_vals, tolerance)


def test_extinction_F99_values():
    # get the correct values
    x, cor_vals, tolerance = get_axav_cor_vals()

    # initialize extinction model
    tmodel = F99()

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=tolerance)


x_vals, axav_vals, tolerance = get_axav_cor_vals()
test_vals = zip(x_vals, axav_vals, np.full(len(x_vals),tolerance))
@pytest.mark.parametrize("test_vals", test_vals)
def test_extinction_F99_single_values(test_vals):
    x, cor_val, tolerance = test_vals

    # initialize extinction model
    tmodel = F99()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val, rtol=tolerance)
