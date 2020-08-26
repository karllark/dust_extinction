import pytest
import numpy as np
import astropy.units as u

from .helpers import grain_models


@pytest.mark.parametrize("model_class", grain_models)
def test_corvals(model_class):
    # instantiate extinction model
    tmodel = model_class()
    # get the possible list of models
    possmodels = tmodel.possnames.keys()

    for cmodel in possmodels:
        tmodel = model_class(cmodel)

        # test array evaluation
        x_vals = tmodel.data_x / u.micron
        y_vals = tmodel.data_axav
        tol = tmodel.data_tolerance
        np.testing.assert_allclose(tmodel(x_vals), y_vals, rtol=tol)


@pytest.mark.parametrize("model_class", grain_models)
def test_extinguish_values_Av_or_Ebv(model_class):
    # just test the default model
    ext = model_class()
    x = np.arange(ext.x_range[0], ext.x_range[1]) / u.micron
    cor_axav = ext(x)

    # calculate the cor_vals in fractional units
    Av = 1.0
    cor_vals_av = np.power(10.0, -0.4 * (cor_axav * Av))
    Ebv = 1.0
    Av_ebv = Ebv * ext.Rv
    cor_vals_ebv = np.power(10.0, -0.4 * cor_axav * Av_ebv)

    # tests
    np.testing.assert_equal(ext.extinguish(x, Av=Av), cor_vals_av)
    np.testing.assert_equal(ext.extinguish(x, Ebv=Ebv), cor_vals_ebv)
