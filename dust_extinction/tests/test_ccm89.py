import numpy as np
import pytest

import astropy.units as u

from ..parameter_averages import CCM89


def get_axav_cor_vals(Rv):
    # testing wavenumbers
    x = np.array(
        [
            10.0,
            9.0,
            8.0,
            7.0,
            6.0,
            5.0,
            4.6,
            4.0,
            3.0,
            2.78,
            2.27,
            1.82,
            1.43,
            1.11,
            0.8,
            0.63,
            0.46,
        ]
    )

    # add units
    x = x / u.micron

    # correct values
    if Rv == 3.1:
        cor_vals = np.array(
            [
                5.23835484,
                4.13406452,
                3.33685933,
                2.77962453,
                2.52195399,
                2.84252644,
                3.18598916,
                2.31531711,
                1.64254927,
                1.56880904,
                1.32257836,
                1.0,
                0.75125994,
                0.47803460,
                0.28206957,
                0.19200814,
                0.11572348,
            ]
        )
    elif Rv == 2.0:
        cor_vals = np.array(
            [
                9.407,
                7.3065,
                5.76223881,
                4.60825807,
                4.01559036,
                4.43845534,
                4.93952892,
                3.39275574,
                2.068771,
                1.9075018,
                1.49999733,
                1.0,
                0.68650255,
                0.36750326,
                0.21678862,
                0.14757062,
                0.08894094,
            ]
        )
    elif Rv == 3.0:
        cor_vals = np.array(
            [
                5.491,
                4.32633333,
                3.48385202,
                2.8904508,
                2.6124774,
                2.9392494,
                3.2922643,
                2.38061642,
                1.66838089,
                1.58933588,
                1.33333103,
                1.0,
                0.74733525,
                0.47133573,
                0.27811315,
                0.18931496,
                0.11410029,
            ]
        )
    elif Rv == 4.0:
        cor_vals = np.array(
            [
                3.533,
                2.83625,
                2.34465863,
                2.03154717,
                1.91092092,
                2.18964643,
                2.46863199,
                1.87454675,
                1.46818583,
                1.43025292,
                1.24999788,
                1.0,
                0.7777516,
                0.52325196,
                0.30877542,
                0.21018713,
                0.12667997,
            ]
        )
    elif Rv == 5.0:
        cor_vals = np.array(
            [
                2.3582,
                1.9422,
                1.66114259,
                1.51620499,
                1.48998704,
                1.73988465,
                1.97445261,
                1.57090496,
                1.3480688,
                1.33480314,
                1.19999799,
                1.0,
                0.79600141,
                0.5544017,
                0.32717278,
                0.22271044,
                0.13422778,
            ]
        )
    elif Rv == 6.0:
        cor_vals = np.array(
            [
                1.575,
                1.34616667,
                1.20546523,
                1.17264354,
                1.20936444,
                1.44004346,
                1.64499968,
                1.36847709,
                1.26799077,
                1.27116996,
                1.16666472,
                1.0,
                0.80816794,
                0.5751682,
                0.33943769,
                0.23105931,
                0.13925965,
            ]
        )
    else:
        cor_vals = np.array([0.0])

    return (x, cor_vals)


@pytest.mark.parametrize("Rv", [2.0, 3.0, 3.1, 4.0, 5.0, 6.0])
def test_extinction_CCM89_values(Rv):
    # get the correct values
    x, cor_vals = get_axav_cor_vals(Rv)

    # initialize extinction model
    tmodel = CCM89(Rv=Rv)

    # test
    np.testing.assert_allclose(tmodel(x), cor_vals)


@pytest.mark.parametrize("Rv", [2.0, 3.0, 3.1, 4.0, 5.0, 6.0])
def test_extinction_CCM89_extinguish_values_Av(Rv):
    # get the correct values
    x, cor_vals = get_axav_cor_vals(Rv)

    # calculate the cor_vals in fractional units
    Av = 1.0
    cor_vals = np.power(10.0, -0.4 * (cor_vals * Av))

    # initialize extinction model
    tmodel = CCM89(Rv=Rv)

    # test
    np.testing.assert_allclose(tmodel.extinguish(x, Av=Av), cor_vals)


@pytest.mark.parametrize("Rv", [2.0, 3.0, 3.1, 4.0, 5.0, 6.0])
def test_extinction_CCM89_extinguish_values_Ebv(Rv):
    # get the correct values
    x, cor_vals = get_axav_cor_vals(Rv)

    # calculate the cor_vals in fractional units
    Ebv = 1.0
    Av = Ebv * Rv
    cor_vals = np.power(10.0, -0.4 * (cor_vals * Av))

    # initialize extinction model
    tmodel = CCM89(Rv=Rv)

    # test
    np.testing.assert_allclose(tmodel.extinguish(x, Ebv=Ebv), cor_vals)


x_vals, axav_vals = get_axav_cor_vals(3.1)
test_vals = zip(x_vals, axav_vals)


@pytest.mark.parametrize("test_vals", test_vals)
def test_extinction_CCM89_single_values(test_vals):
    x, cor_val = test_vals

    # initialize extinction model
    tmodel = CCM89()

    # test
    np.testing.assert_allclose(tmodel(x), cor_val)
    np.testing.assert_allclose(tmodel.evaluate(x, 3.1), cor_val)
