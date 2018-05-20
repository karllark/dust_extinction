import numpy as np
import pytest

import astropy.units as u
from astropy.modeling import InputParameterError

from ..averages import GCC09_MWAvg
from .helpers import _invalid_x_range


x_bad = [-1.0, 0.1, 11., 100.]


@pytest.mark.parametrize("x_invalid", x_bad)
def test_invalid_wavenumbers(x_invalid):
    _invalid_x_range(x_invalid, GCC09_MWAvg(), 'GCC09')


@pytest.mark.parametrize("x_invalid_wavenumber", x_bad/u.micron)
def test_invalid_wavenumbers_imicron(x_invalid_wavenumber):
    _invalid_x_range(x_invalid_wavenumber, GCC09_MWAvg(), 'GCC09')


@pytest.mark.parametrize("x_invalid_micron", u.micron/x_bad)
def test_invalid_micron(x_invalid_micron):
    _invalid_x_range(x_invalid_micron, GCC09_MWAvg(), 'GCC09')


@pytest.mark.parametrize("x_invalid_angstrom", u.angstrom*1e4/x_bad)
def test_invalid_micron(x_invalid_angstrom):
    _invalid_x_range(x_invalid_angstrom, GCC09_MWAvg(), 'GCC09')


def test_extinguish_no_av_or_ebv():
    tmodel = GCC09_MWAvg()
    with pytest.raises(InputParameterError) as exc:
        tmodel.extinguish([1.0])
    assert exc.value.args[0] == 'neither Av or Ebv passed, one required'


def test_extinction_GCC09_values():
    tmodel = GCC09_MWAvg()
    # test
    #  not to numerical precision as we are using the FM90 fits
    #  and spline functions and the correct values are the data
    np.testing.assert_allclose(tmodel(tmodel.obsdata_x),
                               tmodel.obsdata_axav,
                               rtol=tmodel.obsdata_tolerance)


def test_extinction_GCC09_single_values():
    tmodel = GCC09_MWAvg()
    # test
    for x, cor_val in zip(tmodel.obsdata_x, tmodel.obsdata_axav):
        np.testing.assert_allclose(tmodel(x), cor_val,
                                   rtol=tmodel.obsdata_tolerance)


def test_extinction_GCC09_extinguish_values_Av():
    tmodel = GCC09_MWAvg()

    x = np.arange(0.3, 1.0/0.0912, 0.1)/u.micron
    cor_vals = tmodel(x)

    # calculate the cor_vals in fractional units
    Av = 1.0
    cor_vals = np.power(10.0, -0.4*cor_vals*Av)

    # test
    np.testing.assert_equal(tmodel.extinguish(x, Av=Av), cor_vals)


def test_extinction_GCC09_extinguish_values_Ebv():
    tmodel = GCC09_MWAvg()

    x = np.arange(0.3, 1.0/0.0912, 0.1)/u.micron
    cor_vals = tmodel(x)

    # calculate the cor_vals in fractional units
    Ebv = 1.0
    Av = Ebv*tmodel.Rv
    cor_vals = np.power(10.0, -0.4*cor_vals*Av)

    # test
    np.testing.assert_equal(tmodel.extinguish(x, Ebv=Ebv), cor_vals)
