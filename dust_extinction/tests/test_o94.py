import numpy as np
import pytest

import astropy.units as u

from ..parameter_averages import O94


def get_axav_cor_vals(Rv):
    # testing only NIR or UV wavenumbers (optical tested in previous test)
    # O94 is the same as CCM89 for these wavelengths

    x = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.6, 4.0, 0.8, 0.63, 0.46])

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
                0.33943769,
                0.23105931,
                0.13925965,
            ]
        )
    else:
        cor_vals = np.array([0.0])

    return (x, cor_vals)


@pytest.mark.parametrize("Rv", [2.0, 3.0, 3.1, 4.0, 5.0, 6.0])
def test_extinction_O94_extinguish_values_Av(Rv):
    # get the correct values
    x, cor_vals = get_axav_cor_vals(Rv)

    # calculate the cor_vals in fractional units
    Av = 1.0
    cor_vals = np.power(10.0, -0.4 * (cor_vals * Av))

    # initialize extinction model
    tmodel = O94(Rv=Rv)

    # test
    np.testing.assert_allclose(tmodel.extinguish(x, Av=Av), cor_vals)


@pytest.mark.parametrize("Rv", [2.0, 3.0, 3.1, 4.0, 5.0, 6.0])
def test_extinction_O94_extinguish_values_Ebv(Rv):
    # get the correct values
    x, cor_vals = get_axav_cor_vals(Rv)

    # calculate the cor_vals in fractional units
    Ebv = 1.0
    Av = Ebv * Rv
    cor_vals = np.power(10.0, -0.4 * (cor_vals * Av))

    # initialize extinction model
    tmodel = O94(Rv=Rv)

    # test
    np.testing.assert_allclose(tmodel.extinguish(x, Ebv=Ebv), cor_vals)
