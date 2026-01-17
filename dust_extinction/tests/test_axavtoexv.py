import numpy as np
import astropy.units as u

from astropy.modeling.fitting import LevMarLSQFitter

from dust_extinction.shapes import P92
from dust_extinction.conversions import AxAvToExv


def get_axav_cor_vals():

    # Milky Way observed extinction as tabulated by Pei (1992)
    # fmt: off
    MW_x = [0.21, 0.29, 0.45, 0.61, 0.80, 1.11, 1.43, 1.82,
            2.27, 2.50, 2.91, 3.65, 4.00, 4.17, 4.35, 4.57, 4.76,
            5.00, 5.26, 5.56, 5.88, 6.25, 6.71, 7.18, 7.60,
            8.00, 8.50, 9.00, 9.50, 10.00]
    MW_x = np.array(MW_x)
    MW_exvebv = [-3.02, -2.91, -2.76, -2.58, -2.23, -1.60, -0.78, 0.00,
                 1.00, 1.30, 1.80, 3.10, 4.19, 4.90, 5.77, 6.57, 6.23,
                 5.52, 4.90, 4.65, 4.60, 4.73, 4.99, 5.36, 5.91,
                 6.55, 7.45, 8.45, 9.80, 11.30]
    # fmt: on

    MW_exvebv = np.array(MW_exvebv)
    Rv = 3.08
    MW_axav = MW_exvebv / Rv + 1.0

    # add units
    x = MW_x / u.micron

    # correct values
    cor_vals = MW_axav

    return (x, cor_vals)


# @pytest.mark.skip(reason="failing due to an issue with the fitting")
def test_AxAvtoExv_with_P92_fitting():

    # get an observed extinction curve to fit
    x_quant, y_axav = get_axav_cor_vals()
    x = x_quant.value

    # pick an A(V) value
    av = 1.3

    # transform the extinction from AxAv to Exv
    y = (y_axav - 1.0) * av

    # change from defaults to make the best fit harder to find
    p92_init = P92() | AxAvToExv()

    fit = LevMarLSQFitter()
    # accuracy set to avoid warning the fit may have failed
    p92_fit = fit(p92_init, x, y, acc=1e-3)

    fit_vals = p92_fit._parameters

    # fmt: off
    good_vals = [230.06040825320403, 0.04708410695523418, 94.97954884833534, 2.0,
                 25.479966617387454, 0.06387366154554451, 1.2254421564815603,
                 5.077278327831658, 0.05507539590706781, 0.21866103857294794,
                 -1.9493931616922349, 2.0, 0.0, 7.0, -283.2235268628523, 2.0,
                 0.0, 21.0, -8071.411209126977, 2.0, 0.0, 30.0, -6526.471983961995,
                 2.0, 1.304374949048315]
    # fmt: on

    np.testing.assert_allclose(good_vals, fit_vals)
