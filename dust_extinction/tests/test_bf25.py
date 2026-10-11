import numpy as np
import pytest

import astropy.units as u

from dust_extinction.custom_models.bf25 import BF25


def get_axav_cor_vals():
    """
    Reference values taken from bf25_jwst_law.dat
    """
    x = 1.0/np.array(
        [1.20300720, 1.63424360, 1.85088840, 2.12136100, 3.61966750, 4.05162100, 4.70773970, 4.80950540]
    )
    
    x = x/u.micron
    
    cor_vals = np.array(
        [0.32966511, 0.17916582, 0.14136183, 0.10875365, 0.05482474, 0.04443312, 0.04299980, 0.03762482]
    )

    return x, cor_vals


def test_extinction_bf25_values():
    x, cor_vals = get_axav_cor_vals()

    tmodel = BF25()

    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=1e-5)


@pytest.mark.parametrize("x, cor_vals", zip(*get_axav_cor_vals()))
def test_extinction_bf25_single_value(x, cor_vals):
    tmodel = BF25()

    np.testing.assert_allclose(tmodel(x), cor_vals, rtol=1e-5)
