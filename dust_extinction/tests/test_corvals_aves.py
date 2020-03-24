import pytest
import numpy as np

from dust_extinction.averages import (
    RL85_MWAvg,
    G03_SMCBar,
    G03_LMCAvg,
    G03_LMC2,
    GCC09_MWAvg,
)

models = [RL85_MWAvg, G03_SMCBar, G03_LMCAvg, G03_LMC2, GCC09_MWAvg]


@pytest.mark.parametrize("model_class", models)
def test_corvals(model_class):
    # instantiate extinction model
    tmodel = model_class()

    # test array evaluation
    x_vals = tmodel.obsdata_x
    y_vals = tmodel.obsdata_axav
    tol = tmodel.obsdata_tolerance
    np.testing.assert_allclose(tmodel(x_vals), y_vals, rtol=tol)

    # test single value evalutation
    for x, y in zip(x_vals, y_vals):
        np.testing.assert_allclose(tmodel(x), y, rtol=tol)
        np.testing.assert_allclose(tmodel.evaluate(x), y, rtol=tol)
