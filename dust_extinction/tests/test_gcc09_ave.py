import numpy as np

import astropy.units as u

from ..averages import GCC09_MWAvg


def test_extinction_GCC09_extinguish_values_Av():
    tmodel = GCC09_MWAvg()

    x = np.arange(0.3, 1.0 / 0.0912, 0.1) / u.micron
    cor_vals = tmodel(x)

    # calculate the cor_vals in fractional units
    Av = 1.0
    cor_vals = np.power(10.0, -0.4 * cor_vals * Av)

    # test
    np.testing.assert_equal(tmodel.extinguish(x, Av=Av), cor_vals)


def test_extinction_GCC09_extinguish_values_Ebv():
    tmodel = GCC09_MWAvg()

    x = np.arange(0.3, 1.0 / 0.0912, 0.1) / u.micron
    cor_vals = tmodel(x)

    # calculate the cor_vals in fractional units
    Ebv = 1.0
    Av = Ebv * tmodel.Rv
    cor_vals = np.power(10.0, -0.4 * cor_vals * Av)

    # test
    np.testing.assert_equal(tmodel.extinguish(x, Ebv=Ebv), cor_vals)
