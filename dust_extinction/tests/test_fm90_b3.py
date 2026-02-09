import numpy as np

import astropy.units as u
from dust_extinction.shapes import _curve_F99_method, FM90_B3


def test_curve_F99_method_fm90_B3():
    """Test that _curve_F99_method works with fm90_version='B3'"""
    
    # Test parameters that should trigger the B3 branch
    x = np.array([4.0, 5.0, 6.0])  # UV wavelengths
    Rv = 3.1
    C1 = 0.05
    C2 = 0.1
    bump_param = 2.0  # B3 parameter
    C4 = 0.4
    xo = 4.6
    gamma = 1.0
    optnir_axav_x = np.array([1.0, 2.0])
    optnir_axav_y = np.array([0.5, 1.0])
    fm90_version = "B3"
    
    # This should execute line 100
    result = _curve_F99_method(
        x, Rv, C1, C2, bump_param, C4, xo, gamma,
        optnir_axav_x, optnir_axav_y, fm90_version
    )
    
    # Verify it returns expected shape
    assert len(result) == len(x)
    assert np.all(np.isfinite(result))
    
    # Compare with direct FM90_B3 model to ensure consistency
    fm90_b3_model = FM90_B3(C1=C1, C2=C2, B3=bump_param, C4=C4, xo=xo, gamma=gamma)
    expected_fm90_part = fm90_b3_model(x / u.micron) / Rv + 1.0
    # Should be approximately equal for UV part
    assert np.allclose(result, expected_fm90_part, rtol=1e-10)