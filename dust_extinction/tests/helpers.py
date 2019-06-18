import pytest


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
