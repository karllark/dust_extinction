import numpy as np
import pytest

import astropy.units as u
from astropy.modeling.fitting import LevMarLSQFitter

from ..shapes import P92
from .helpers import _invalid_x_range

x_bad = [-1.0, 1001.]


@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    _invalid_x_range(x_invalid, P92(), 'P92')


@pytest.mark.parametrize("x_invalid_wavenumber", x_bad/u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    _invalid_x_range(x_invalid_wavenumber, P92(), 'P92')


@pytest.mark.parametrize("x_invalid_micron", u.micron/x_bad)
def test_invalid_micron(x_invalid_micron):
    _invalid_x_range(x_invalid_micron, P92(), 'P92')


@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom*1e4/x_bad)
def test_invalid_micron(x_invalid_angstrom):
    _invalid_x_range(x_invalid_angstrom, P92(), 'P92')


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
    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=0.25, atol=0.01)


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

    good_vals = [218.957451206, 0.0481323587043, 89.8639079339, 2.0,
                 19.8918861271, 0.0674934514694, 0.919702726068, 5.1217891448,
                 0.0548568919776, 0.218664938289, -1.9496661308, 2.0,
                 0.0, 13.0, 38.279150331, 2.0,
                 0.0, 15.0, -76.7467816812, 2.0,
                 0.0, 20.0, -2508.60124085, 2.0]

    np.testing.assert_allclose(good_vals, fit_vals)
