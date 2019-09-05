import numpy as np
import pytest

import astropy.units as u
from astropy.modeling import InputParameterError

from ..averages import G03_SMCBar, G03_LMCAvg, G03_LMC2

models = [G03_SMCBar(), G03_LMCAvg(), G03_LMC2()]


@pytest.mark.parametrize("tmodel", models)
def test_extinguish_no_av_or_ebv(tmodel):
    with pytest.raises(InputParameterError) as exc:
        tmodel.extinguish([1.0])
    assert exc.value.args[0] == "neither Av or Ebv passed, one required"


@pytest.mark.parametrize("tmodel", models)
def test_extinction_G03_values(tmodel):
    # test
    #  not to numerical precision as we are using the FM90 fits
    #  and spline functions and the correct values are the data
    np.testing.assert_allclose(
        tmodel(tmodel.obsdata_x), tmodel.obsdata_axav, rtol=tmodel.obsdata_tolerance
    )


@pytest.mark.parametrize("tmodel", models)
def test_extinction_G03_single_values(tmodel):
    # test
    for x, cor_val in zip(tmodel.obsdata_x, tmodel.obsdata_axav):
        np.testing.assert_allclose(tmodel(x), cor_val, rtol=tmodel.obsdata_tolerance)
        np.testing.assert_allclose(
            tmodel.evaluate(x), cor_val, rtol=tmodel.obsdata_tolerance
        )


@pytest.mark.parametrize("tmodel", models)
def test_extinction_G03_extinguish_values_Av(tmodel):
    x = np.arange(0.3, 10.0, 0.1) / u.micron
    cor_vals = tmodel(x)

    # calculate the cor_vals in fractional units
    Av = 1.0
    cor_vals = np.power(10.0, -0.4 * (cor_vals * Av))

    # test
    np.testing.assert_equal(tmodel.extinguish(x, Av=Av), cor_vals)


@pytest.mark.parametrize("tmodel", models)
def test_extinction_G03_extinguish_values_Ebv(tmodel):
    x = np.arange(0.3, 10.0, 0.1) / u.micron
    cor_vals = tmodel(x)

    # calculate the cor_vals in fractional units
    Ebv = 1.0
    Av = Ebv * tmodel.Rv
    cor_vals = np.power(10.0, -0.4 * cor_vals * Av)

    # test
    np.testing.assert_equal(tmodel.extinguish(x, Ebv=Ebv), cor_vals)
