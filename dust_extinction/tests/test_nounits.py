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


@pytest.mark.parametrize("model", [CCM89, O94, F99, F04, VCG04, GCC09, M14, G16, F20])
def test_nounits_warning(model):
    ext = model()
    x = np.arange(ext.x_range[0], ext.x_range[1], 0.1)

    with pytest.warns(
        UserWarning, match="x has no units, assuming x units are inverse microns"
    ):
        ext(x)


@pytest.mark.parametrize("model", [CCM89, O94, F99, F04, VCG04, GCC09, M14, G16, F20])
def test_units_nowarning_expected(model):
    ext = model()
    x = [0.5 * (ext.x_range[0] + ext.x_range[1])] / u.micron
    # x = np.arange(ext.x_range[0], ext.x_range[1], 0.1) * u.micron

    with pytest.warns(None) as record:
        ext(x)
    assert len(record) == 0
