import numpy as np
import astropy.units as u

from astropy.modeling.fitting import LevMarLSQFitter

from ..shapes import P92
from ..conversions import AxAvToExv


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

    return (MW_x/u.micron, MW_axav)


class P92_Exv(P92 | AxAvToExv):
    """
    Evalute P92 on E(x-V) data including solving for A(V)
    """


def test_AxAvtoExv_with_P92_fitting():

    # get an observed extinction curve to fit
    x_quant, y_axav = get_axav_cor_vals()
    x = x_quant.value

    # pick an A(V) value
    av = 1.3

    # transform the extinction from AxAv to Exv
    y = (y_axav - 1.0)*av

    # change from defaults to make the best fit harder to find
    p92_init = P92_Exv()

    fit = LevMarLSQFitter()
    # accuracy set to avoid warning the fit may have failed
    p92_fit = fit(p92_init, x, y, acc=1e-3)

    fit_vals = p92_fit._parameters

    good_vals = [218.353426543, 0.0483364461588, 90.1590593157, 2.0,
                 19.3315222458, 0.0672874204509, 0.777521482345, 5.05521692376,
                 0.0550781710089, 0.218663831017, -1.94939455863, 2.0,
                 0.0, 13.0, 29.389593976, 2.0,
                 0.0, 21.0, 927.437855498, 2.0,
                 0.0, 30.0, 777.989416796, 2.0,
                 1.3044133337]

    np.testing.assert_allclose(good_vals, fit_vals)
