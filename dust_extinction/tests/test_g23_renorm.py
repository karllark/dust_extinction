import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import G23


def test_G23_renorm_false():
    """Test that G23 with renorm=False sets normval to 1.0"""
    # Test with renorm=False (should execute line 1460)
    model = G23(Rv=3.1, renorm=False)

    # Verify normval is set to 1.0 (line 1460)
    assert model.normval == 1.0

    # Test that it produces different results than renorm=True
    model_renorm_true = G23(Rv=3.1, renorm=True)
    assert model_renorm_true.normval == 0.9854

    # Verify the models actually produce different extinction values
    x = np.array([1.0, 2.0, 3.0]) / u.micron
    result_false = model(x)
    result_true = model_renorm_true(x)

    # renorm=False should give different values than renorm=True
    assert not np.allclose(result_false, result_true)
