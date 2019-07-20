import pytest

import numpy as np

import astropy.units as u

from dust_extinction.parameter_averages import (
    CCM89,
    O94,
    F99,
    F04,
    VCG04,
    GCC09,
    M14,
    G16,
    F20,
)
from dust_extinction.shapes import FM90, P92
from dust_extinction.averages import G03_SMCBar, G03_LMCAvg, G03_LMC2, GCC09_MWAvg

all_models = [
    CCM89,
    O94,
    F99,
    F04,
    VCG04,
    GCC09,
    M14,
    G16,
    F20,
    FM90,
    P92,
    G03_SMCBar,
    G03_LMCAvg,
    G03_LMC2,
    GCC09_MWAvg,
]


@pytest.mark.parametrize("model", all_models)
def test_nounits_warning(model):
    ext = model()
    x = np.arange(ext.x_range[0], ext.x_range[1], 0.1)

    with pytest.warns(
        UserWarning, match="x has no units, assuming x units are inverse microns"
    ):
        ext(x)


@pytest.mark.parametrize("model", all_models)
def test_units_nowarning_expected(model):
    ext = model()
    x = [0.5 * (ext.x_range[0] + ext.x_range[1])] / u.micron

    with pytest.warns(None) as record:
        ext(x)
    assert len(record) == 0
