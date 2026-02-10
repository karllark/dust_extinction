import numpy as np

import astropy.units as u
from astropy.modeling.fitting import LevMarLSQFitter
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
        x,
        Rv,
        C1,
        C2,
        bump_param,
        C4,
        xo,
        gamma,
        optnir_axav_x,
        optnir_axav_y,
        fm90_version,
    )

    # Verify it returns expected shape
    assert len(result) == len(x)
    assert np.all(np.isfinite(result))

    # Compare with direct FM90_B3 model to ensure consistency
    fm90_b3_model = FM90_B3(C1=C1, C2=C2, B3=bump_param, C4=C4, xo=xo, gamma=gamma)
    expected_fm90_part = fm90_b3_model(x / u.micron) / Rv + 1.0
    # Should be approximately equal for UV part
    assert np.allclose(result, expected_fm90_part, rtol=1e-10)


def test_FM90_B3_fitting():
    """Test that FM90_B3 can be fitted with analytical derivatives"""

    # Generate test data
    np.random.seed(42)
    x_data = np.linspace(3.5, 8.0, 15)

    # True parameters
    true_params = {"C1": 0.05, "C2": 0.1, "B3": 2.0, "C4": 0.3, "xo": 4.6, "gamma": 1.0}

    # Generate synthetic data
    model = FM90_B3(**true_params)
    y_true = model(x_data)
    y_data = y_true + np.random.normal(0, 0.02, len(x_data))

    # Fit using analytical derivatives
    fitter = LevMarLSQFitter()
    model_fit = FM90_B3()
    fitted_model = fitter(model_fit, x_data, y_data)

    # Verify fit converged to reasonable values
    for param_name, true_value in true_params.items():
        fitted_value = getattr(fitted_model, param_name).value
        # Allow tolerance due to noise
        assert np.isclose(
            fitted_value, true_value, rtol=0.3, atol=0.15
        ), f"{param_name}: fitted {fitted_value:.3f}, true {true_value:.3f}"
