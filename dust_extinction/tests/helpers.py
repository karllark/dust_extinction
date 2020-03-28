import pytest

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
    RL85_MWGC,
    G03_SMCBar,
    G03_LMCAvg,
    G03_LMC2,
    I05_MWAvg,
    CT06_MWGC,
    CT06_MWLoc,
    GCC09_MWAvg,
    F11_MWGC,
)

param_ave_models_Rv = [CCM89, O94, F99, F04, VCG04, GCC09, M14, F19]
param_ave_models_Rv_fA = [G16]
param_ave_models = param_ave_models_Rv + param_ave_models_Rv_fA
shape_models = [FM90, P92]
ave_models = [
    RL85_MWGC,
    G03_SMCBar,
    G03_LMCAvg,
    G03_LMC2,
    I05_MWAvg,
    CT06_MWGC,
    CT06_MWLoc,
    GCC09_MWAvg,
    F11_MWGC,
]

all_models = param_ave_models_Rv + param_ave_models_Rv_fA + shape_models + ave_models


def _invalid_x_range(x, tmodel, modname):
    with pytest.raises(ValueError) as exc:
        tmodel(x)
    assert (
        exc.value.args[0]
        == "Input x outside of range defined for "
        + modname
        + " ["
        + str(tmodel.x_range[0])
        + " <= x <= "
        + str(tmodel.x_range[1])
        + ", x has units 1/micron]"
    )
