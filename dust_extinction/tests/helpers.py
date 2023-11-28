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
    D22,
    G23,
)
from dust_extinction.shapes import FM90, FM90_B3, P92, G21
from dust_extinction.averages import (
    RL85_MWGC,
    RRP89_MWGC,
    B92_MWAvg,
    G03_SMCBar,
    G03_LMCAvg,
    G03_LMC2,
    I05_MWAvg,
    CT06_MWGC,
    CT06_MWLoc,
    GCC09_MWAvg,
    F11_MWGC,
    G21_MWAvg,
    D22_MWAvg,
)
from dust_extinction.grain_models import DBP90, WD01, D03, ZDA04, C11, J13, HD23

param_ave_models_Rv = [CCM89, O94, F99, F04, VCG04, GCC09, M14, F19, D22, G23]
param_ave_models_Rv_fA = [G16]
param_ave_models = param_ave_models_Rv + param_ave_models_Rv_fA
shape_models = [FM90, FM90_B3, P92, G21]
ave_models = [
    RL85_MWGC,
    RRP89_MWGC,
    B92_MWAvg,
    G03_SMCBar,
    G03_LMCAvg,
    G03_LMC2,
    I05_MWAvg,
    CT06_MWGC,
    CT06_MWLoc,
    GCC09_MWAvg,
    F11_MWGC,
    G21_MWAvg,
    D22_MWAvg,
]
grain_models = [DBP90, WD01, D03, ZDA04, C11, J13, HD23]

all_models = (
    param_ave_models_Rv
    + param_ave_models_Rv_fA
    + shape_models
    + ave_models
    + grain_models
)


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
