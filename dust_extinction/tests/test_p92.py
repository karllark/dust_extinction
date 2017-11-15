import numpy as np
import pytest

import astropy.units as u
from astropy.modeling import InputParameterError
from astropy.modeling.fitting import LevMarLSQFitter

from ..dust_extinction import P92

x_bad = [-1.0, 1001.]
@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    tmodel = P92()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid)
    assert exc.value.args[0] == 'Input x outside of range defined for P92' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_wavenumber", x_bad/u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    tmodel = P92()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_wavenumber)
    assert exc.value.args[0] == 'Input x outside of range defined for P92' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_micron", u.micron/x_bad)
def test_invalid_micron(x_invalid_micron):
    tmodel = P92()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_micron)
    assert exc.value.args[0] == 'Input x outside of range defined for P92' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom*1e4/x_bad)
def test_invalid_micron(x_invalid_angstrom):
    tmodel = P92()
    with pytest.raises(ValueError) as exc:
        tmodel(x_invalid_angstrom)
    assert exc.value.args[0] == 'Input x outside of range defined for P92' \
                                + ' [' \
                                + str(tmodel.x_range[0]) \
                                +  ' <= x <= ' \
                                + str(tmodel.x_range[1]) \
                                + ', x has units 1/micron]'

def get_axav_cor_vals():

    # Milky Way observed extinction as tabulated by Pei (1992)
    MW_x = [0.21, 0.29, 0.45, 0.61, 0.80, 1.11, 1.43, 1.82,
            2.27, 2.50, 2.91, 3.65, 4.00, 4.17, 4.35, 4.57, 4.76,
            5.00, 5.26, 5.56, 5.88, 6.25, 6.71, 7.18, 7.60,
            8.00, 8.50, 9.00, 9.50, 10.00]
    MW_x = np.array(MW_x)
    MW_exvebv = [-3.02, -2.91, -2.76, -2.58, -2.23, -1.60, -0.78, 0.00,
                 1.00, 1.30, 1.80, 3.10, 4.19, 4.90, 5.77, 6.57, 6.23,
                 5.52, 4.90, 4.65, 4.60, 4.73, 4.99, 5.36, 5.91,
                 6.55, 7.45, 8.45, 9.80, 11.30]
    MW_exvebv = np.array(MW_exvebv)
    Rv = 3.08
    MW_axav = MW_exvebv/Rv + 1.0


    # add units
    x = MW_x/u.micron

    # correct values
    cor_vals = MW_axav

    return (x, cor_vals)

def test_extinction_P92_values():
    # get the correct values
    x, cor_vals = get_axav_cor_vals()

    # initialize extinction model
    tmodel = P92()

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=0.1, atol=0.01)


def test_P92_fitting():

    # get an observed extinction curve to fit
    x_quant, y = get_axav_cor_vals()

    x = x_quant.value

    # change from defaults to make the best fit harder to find
    p92_init = P92()

    fit = LevMarLSQFitter()
    # accuracy set to avoid warning the fit may have failed
    p92_fit = fit(p92_init, x, y, acc=1e-3)

    fit_vals = p92_fit._parameters

    good_vals = [222.820720554, 0.0468116978742, 87.5683270158, 2.0,
                 18.1622695074, 0.0770527204445, 2.83090815471, 6.40496987921,
                 0.0518566585953, 0.218641802256, -1.95087797308, 2.0,
                 0.00600017900912, 13.0, 109.369888618, 2.0,
                 0.09441454528, 15.0, -251.33735749, 2.0,
                 0.189219789029, 20.0, -1130.90875587, 2.0]

    np.testing.assert_allclose(good_vals, fit_vals)
