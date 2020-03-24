import numpy as np
import pytest

import astropy.units as u
from astropy.modeling.fitting import LevMarLSQFitter

from ..averages import G03_LMCAvg
from ..shapes import FM90


def get_elvebv_cor_vals():
    # testing wavenumbers
    x = np.array([3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 9.0, 10.0])

    # add units
    x = x / u.micron

    # correct values
    cor_vals = np.array(
        [
            2.9829317,
            4.1215415,
            6.4135842,
            5.6574243,
            4.7573250,
            5.4905843,
            9.2853567,
            12.462238,
        ]
    )

    return (x, cor_vals)


def test_extinction_FM90_values():
    # get the correct values
    x, cor_vals = get_elvebv_cor_vals()

    # initialize extinction model
    tmodel = FM90()

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals)


x_vals, axav_vals = get_elvebv_cor_vals()
test_vals = zip(x_vals, axav_vals)


@pytest.mark.parametrize("xtest_vals", test_vals)
def test_extinction_FM90_single_values(xtest_vals):
    x, cor_val = xtest_vals

    # initialize extinction model
    tmodel = FM90()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val)
    np.testing.assert_allclose(
        tmodel.evaluate(
            x,
            FM90.C1.default,
            FM90.C2.default,
            FM90.C3.default,
            FM90.C4.default,
            FM90.xo.default,
            FM90.gamma.default,
        ),
        cor_val,
    )


def test_FM90_fitting():

    # get an observed extinction curve to fit
    g03_model = G03_LMCAvg()

    x = g03_model.obsdata_x
    # convert to E(x-V)/E(B0V)
    y = (g03_model.obsdata_axav - 1.0) * g03_model.Rv
    # only fit the UV portion (FM90 only valid in UV)
    gindxs, = np.where(x > 3.125)

    fm90_init = FM90()
    fit = LevMarLSQFitter()
    g03_fit = fit(fm90_init, x[gindxs], y[gindxs])
    fit_vals = [
        g03_fit.C1.value,
        g03_fit.C2.value,
        g03_fit.C3.value,
        g03_fit.C4.value,
        g03_fit.xo.value,
        g03_fit.gamma.value,
    ]

    good_vals = np.array(
        [
            -0.958016797002,
            1.0109751831,
            2.96430606652,
            0.313137860902,
            4.59996300532,
            0.99000982258,
        ]
    )

    np.testing.assert_allclose(good_vals, fit_vals)
