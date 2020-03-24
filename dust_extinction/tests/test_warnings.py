import pytest

import numpy as np

import astropy.units as u
from astropy.modeling import InputParameterError

from .helpers import _invalid_x_range
from dust_extinction.parameter_averages import (
    CCM89,
    O94,
    F99,
    F04,
    VCG04,
    GCC09,
    M14,
    G16,
    F19,
)
from dust_extinction.shapes import FM90, P92
from dust_extinction.averages import (
    RL85_MWAvg,
    G03_SMCBar,
    G03_LMCAvg,
    G03_LMC2,
    GCC09_MWAvg,
)

param_ave_models_Rv = [CCM89, O94, F99, F04, VCG04, GCC09, M14, F19]
param_ave_models_Rv_fA = [G16]
param_ave_models = param_ave_models_Rv + param_ave_models_Rv_fA
shape_models = [FM90, P92]
ave_models = [RL85_MWAvg, G03_SMCBar, G03_LMCAvg, G03_LMC2, GCC09_MWAvg]

all_models = param_ave_models_Rv + param_ave_models_Rv_fA + shape_models + ave_models

x_bad = [-1.0, 13.0, 100.0]


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


@pytest.mark.parametrize("model", param_ave_models_Rv)
@pytest.mark.parametrize("Rv_invalid", [-1.0, 0.0, 1.9, 6.1, 10.0])
def test_invalid_Rv_input(model, Rv_invalid):
    with pytest.raises(InputParameterError) as exc:
        model(Rv=Rv_invalid)
    assert exc.value.args[0] == "parameter Rv must be between 2.0 and 6.0"


@pytest.mark.parametrize("RvA_invalid", [-1.0, 0.0, 1.9, 6.1, 10.0])
@pytest.mark.parametrize("model", param_ave_models_Rv_fA)
def test_invalid_RvA_input(model, RvA_invalid):
    with pytest.raises(InputParameterError) as exc:
        model(RvA=RvA_invalid)
    assert exc.value.args[0] == "parameter RvA must be between 2.0 and 6.0"


@pytest.mark.parametrize("fA_invalid", [-1.0, -0.1, 1.1, 10.0])
@pytest.mark.parametrize("model", param_ave_models_Rv_fA)
def test_invalid_fA_input(model, fA_invalid):
    with pytest.raises(InputParameterError) as exc:
        model(fA=fA_invalid)
    assert exc.value.args[0] == "parameter fA must be between 0.0 and 1.0"


@pytest.mark.parametrize("model", param_ave_models + ave_models)
@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(model, x_invalid):
    tmodel = model()
    _invalid_x_range(x_invalid, tmodel, tmodel.__class__.__name__)
    _invalid_x_range(x_invalid / u.micron, tmodel, tmodel.__class__.__name__)
    _invalid_x_range(u.micron / x_invalid, tmodel, tmodel.__class__.__name__)
    _invalid_x_range(u.angstrom * 1e4 / x_invalid, tmodel, tmodel.__class__.__name__)


@pytest.mark.parametrize("model", param_ave_models + ave_models)
def test_extinguish_no_av_or_ebv(model):
    ext = model()
    with pytest.raises(InputParameterError) as exc:
        ext.extinguish(ext.x_range[0])
    assert exc.value.args[0] == "neither Av or Ebv passed, one required"
