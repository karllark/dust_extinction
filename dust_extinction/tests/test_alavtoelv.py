import numpy as np
import astropy.units as u

from astropy.modeling.fitting import LevMarLSQFitter

from ..dust_extinction import (P92, AlAvToElv)


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


class P92_Elv(P92 | AlAvToElv):
    """
    Evalute P92 on E(l-V) data including solving for A(V)
    """


def test_P92_fitting():

    # get an observed extinction curve to fit
    x_quant, y_axav = get_axav_cor_vals()
    x = x_quant.value

    # pick an A(V) value
    av = 1.3

    # transform the extinction from AxAv to Exv
    y = (y_axav - 1.0)*av

    # change from defaults to make the best fit harder to find
    p92_init = P92_Elv()

    fit = LevMarLSQFitter()
    # accuracy set to avoid warning the fit may have failed
    p92_fit = fit(p92_init, x, y, acc=1e-3)

    fit_vals = p92_fit._parameters

    good_vals = [220.806743348, 0.0474488526572, 88.4203515994, 2.0,
                 17.0089730125, 0.0729612929387, 1.38876455935, 5.63200040759,
                 0.052051591862, 0.218617616831, -1.95160129592, 2.0,
                 0.0, 7.0, -278.831631235, 2.0,
                 0.0, 21.0, -6617.22039032, 2.0,
                 0.0, 30.0, -4992.15823779, 2.0,
                 1.29867067922]

    np.testing.assert_allclose(good_vals, fit_vals)
