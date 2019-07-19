import pytest

from dust_extinction.parameter_averages import F99


@pytest.mark.parametrize("model", [F99])
def test_nounits_warning(model):
    ext = model()


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


import numpy as np
from dust_extinction.parameter_averages import F99
import astropy.units as u
import matplotlib.pyplot as plt

wav = np.arange(0.1, 3.0, 0.001) * u.micron
for model in [F99]:
    for R in (2.0, 3.0, 4.0):
        # Initialize the extinction model
        ext = model(Rv=R)
        plt.plot(1 / wav, ext(wav), label=model.name + " R=" + str(R))
