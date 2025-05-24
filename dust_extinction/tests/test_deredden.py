import pytest
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u

from dust_extinction.parameter_averages import (
    CCM89,
    F99,
)
from dust_extinction.averages import G03_SMCBar  # A non-RV dependent model
from dust_extinction.deredden import deredden_flux
from dust_extinction.baseclasses import BaseExtModel, BaseExtRvModel


# --- Dummy Models for specific test cases ---
class DummyRvModel(BaseExtRvModel):
    Rv_range = [2.0, 6.0]  # Overriding the range for the dummy, if necessary
    x_range = [0.1, 10.0]  # Specify valid range for this dummy model

    def __init__(self, Rv=3.1):
        # BaseExtRvModel's __init__ does not take Rv.
        # It inherits from BaseExtModel, whose __init__ takes no model-specific args.
        # The Rv parameter is defined on BaseExtRvModel.
        # Setting self.Rv here uses the Parameter descriptor.
        super().__init__()
        self.Rv = Rv

    def evaluate(self, x_in, Rv):
        # When the model is called (e.g., model(x)), astropy's modeling framework
        # takes parameters from the model's param_sets (like self.Rv) and passes
        # them as keyword arguments to evaluate. So, 'Rv' here will be the
        # value of the Rv parameter of the model instance.
        return x_in / Rv


class DummyNonRvModel(BaseExtModel):
    x_range = [0.1, 10.0]  # 1/micron
    # No Rv parameter in __init__ or as attribute

    def evaluate(self, x_in, **kwargs):
        # A(x)/A(V) = x (simple, not physical, for testing)
        return x_in


# --- Fixtures ---
@pytest.fixture
def basic_inputs():
    """Provide basic inputs (no units) for testing."""
    wavelengths = np.array([0.4, 0.5, 0.6])  # microns
    flux = np.array([1.0, 1.5, 2.0])
    # Matched to the expected values from previous successful test runs for CCM89, Rv=3.1, Ebv=0.2
    # Av = 0.62
    # Original flux before reddening for these tests was approx:
    # 1.0 / 2.307862 = 0.4333
    # 1.5 / 2.847138 = 0.5268
    # 2.0 / 3.356565 = 0.5959
    return wavelengths, flux, {"ebv": 0.2, "rv": 3.1}, {"av": 0.62}


@pytest.fixture
def basic_inputs_units():
    """Provide basic inputs with astropy units."""
    wavelengths = np.array([0.4, 0.5, 0.6]) * u.micron
    flux = np.array([1.0, 1.5, 2.0]) * u.Jy
    return (
        wavelengths,
        flux,
        {"ebv": 0.2, "rv": 3.1},
        {"av": 0.62 * u.mag},
    )  # Av can have mag units


# --- Tests for deredden_flux ---


def test_deredden_flux_ebv_rv_dependent_no_units(basic_inputs):
    """Test deredden_flux with E(B-V) for R(V)-dependent model (CCM89), no units."""
    wavelengths, flux, params_ebv, _ = basic_inputs
    dered_flux_val = deredden_flux(
        wavelengths, flux, model_class=CCM89, ebv=params_ebv["ebv"], rv=params_ebv["rv"]
    )
    assert isinstance(dered_flux_val, np.ndarray)
    # Expected values from previous successful test runs with CCM89, Rv=3.1, Ebv=0.2
    expected_dered_flux = np.array([2.307862, 2.847138, 3.356565])
    assert_allclose(dered_flux_val, expected_dered_flux, rtol=1e-5)


def test_deredden_flux_ebv_rv_dependent_with_units(basic_inputs_units):
    """Test deredden_flux with E(B-V) for R(V)-dependent model (CCM89), with units."""
    wavelengths, flux, params_ebv, _ = basic_inputs_units
    dered_flux_q = deredden_flux(
        wavelengths, flux, model_class=CCM89, ebv=params_ebv["ebv"], rv=params_ebv["rv"]
    )
    assert isinstance(dered_flux_q, u.Quantity)
    assert dered_flux_q.unit == flux.unit
    expected_dered_flux_values = np.array([2.307862, 2.847138, 3.356565])
    assert_allclose(dered_flux_q.value, expected_dered_flux_values, rtol=1e-5)


def test_deredden_flux_av_rv_dependent_with_units(basic_inputs_units):
    """Test deredden_flux with A(V) for R(V)-dependent model (F99), with units."""
    wavelengths, flux, _, params_av = basic_inputs_units
    # Using F99 to show av works with other Rv models too
    dered_flux_q = deredden_flux(wavelengths, flux, model_class=F99, av=params_av["av"])
    assert isinstance(dered_flux_q, u.Quantity)
    assert dered_flux_q.unit == flux.unit
    # Expected values for F99, Av=0.62 at 0.4, 0.5, 0.6 micron
    # F99 model, Rv=3.1 (default for F99 if not specified, deredden_flux passes rv=3.1 by default)
    # A(V) = 0.62
    # model = F99(Rv=3.1)
    # wl_inv_micron = 1.0 / wavelengths.to_value(u.micron)
    # axav = model(wl_inv_micron)
    # alambda = axav * 0.62
    # expected_dered_flux = flux.value * np.power(10.0, 0.4 * alambda)
    expected_dered_flux_values = np.array([2.254119, 2.846173, 3.281264])
    assert_allclose(dered_flux_q.value, expected_dered_flux_values, rtol=1e-5)


def test_deredden_flux_av_non_rv_dependent_model(basic_inputs_units):
    """Test deredden_flux with A(V) for a non-R(V)-dependent model (G03_SMCBar)."""
    wavelengths, flux, _, params_av = basic_inputs_units
    # G03_SMCBar is an average model, does not take Rv
    dered_flux_q = deredden_flux(
        wavelengths, flux, model_class=G03_SMCBar, av=params_av["av"]
    )
    assert isinstance(dered_flux_q, u.Quantity)
    assert dered_flux_q.unit == flux.unit
    # Check it runs and produces different results from CCM89 due to different model
    ccm89_flux = deredden_flux(wavelengths, flux, model_class=CCM89, av=params_av["av"])
    assert not np.allclose(dered_flux_q.value, ccm89_flux.value)
    # Expected for G03_SMCBar, Av=0.62
    # model = G03_SMCBar()
    # ...
    expected_dered_flux_values = np.array([2.412804, 2.886646, 3.326536])
    assert_allclose(dered_flux_q.value, expected_dered_flux_values, rtol=1e-5)


def test_deredden_flux_ebv_non_rv_dependent_error(basic_inputs_units):
    """Test error when using E(B-V) with a non-R(V)-dependent model."""
    wavelengths, flux, params_ebv, _ = basic_inputs_units
    with pytest.raises(
        ValueError,
        match="Model G03_SMCBar is not R(V)-dependent. A(V) must be provided",
    ):
        deredden_flux(
            wavelengths,
            flux,
            model_class=G03_SMCBar,
            ebv=params_ebv["ebv"],
            rv=params_ebv["rv"],
        )


def test_deredden_flux_different_rv_explicit(basic_inputs_units):
    """Test deredden_flux with an explicit non-default Rv value."""
    wavelengths, flux, params_ebv, _ = basic_inputs_units
    rv_custom = 5.0
    dered_flux_rv5 = deredden_flux(
        wavelengths, flux, model_class=CCM89, ebv=params_ebv["ebv"], rv=rv_custom
    )
    dered_flux_rv31 = deredden_flux(
        wavelengths, flux, model_class=CCM89, ebv=params_ebv["ebv"], rv=3.1
    )
    assert isinstance(dered_flux_rv5, u.Quantity)
    assert np.all(
        dered_flux_rv5.value > dered_flux_rv31.value
    )  # Higher Rv -> more extinction correction


def test_round_trip_rv_model(basic_inputs_units):
    """Test redden then deredden with an R(V)-dependent model (CCM89)."""
    wavelengths, original_flux, params_ebv, _ = basic_inputs_units
    rv = params_ebv["rv"]
    ebv = params_ebv["ebv"]

    model = CCM89(Rv=rv)
    attenuation_factor = model.extinguish(wavelengths, Ebv=ebv)
    reddened_flux = original_flux * attenuation_factor

    dereddened_flux = deredden_flux(
        wavelengths, reddened_flux, model_class=CCM89, ebv=ebv, rv=rv
    )
    assert_allclose(dereddened_flux.value, original_flux.value, rtol=1e-6)


def test_round_trip_non_rv_model_av(basic_inputs_units):
    """Test redden then deredden with a non-R(V)-model using A(V)."""
    wavelengths, original_flux, _, params_av = basic_inputs_units
    av = params_av["av"]

    model = G03_SMCBar()  # Non-Rv model
    attenuation_factor = model.extinguish(
        wavelengths, Av=av.value if isinstance(av, u.Quantity) else av
    )
    reddened_flux = original_flux * attenuation_factor

    dereddened_flux = deredden_flux(
        wavelengths, reddened_flux, model_class=G03_SMCBar, av=av
    )
    assert_allclose(dereddened_flux.value, original_flux.value, rtol=1e-6)


def test_parameter_errors(basic_inputs_units):
    """Test errors for missing or invalid av/ebv parameters."""
    wavelengths, flux, _, _ = basic_inputs_units
    with pytest.raises(ValueError, match="Either 'av' or 'ebv' must be provided."):
        deredden_flux(wavelengths, flux, model_class=CCM89)
    with pytest.raises(ValueError, match="A(V) = -0.1 must be non-negative."):
        deredden_flux(wavelengths, flux, model_class=CCM89, av=-0.1)
    with pytest.raises(ValueError, match="E(B-V) = -0.1 must be non-negative."):
        deredden_flux(wavelengths, flux, model_class=CCM89, ebv=-0.1, rv=3.1)


def test_wavelength_out_of_range_ccm89(basic_inputs_units):
    """Test wavelength out of range for CCM89."""
    wavelengths_short = np.array([0.01]) * u.micron
    wavelengths_long = np.array([10.0]) * u.micron
    _, flux, _, params_av = basic_inputs_units

    # Using CCM89's known range [0.3, 10.0] 1/micron -> [0.1, 3.33] micron
    with pytest.raises(ValueError, match=r"Input x outside of range defined for CCM89"):
        deredden_flux(wavelengths_short, flux[0], model_class=CCM89, av=params_av["av"])
    with pytest.raises(ValueError, match=r"Input x outside of range defined for CCM89"):
        deredden_flux(wavelengths_long, flux[0], model_class=CCM89, av=params_av["av"])


def test_invalid_model_class_type():
    """Test passing a non-model class type."""
    with pytest.raises(
        TypeError, match="model_class must be a subclass of BaseExtModel."
    ):
        deredden_flux(0.5 * u.micron, 1.0 * u.Jy, model_class=str, av=0.1)


def test_av_precedence(basic_inputs_units):
    """Test that A(V) takes precedence if both A(V) and E(B-V) are given."""
    wavelengths, flux, params_ebv, params_av = basic_inputs_units
    # Use CCM89, provide Av and also Ebv/Rv that would give a *different* Av
    av_priority = params_av["av"]  # 0.62
    ebv_ignored = 0.1  # This would give Av = 0.31 if Rv=3.1
    rv_ignored = 3.1

    # Calculate expected flux if only av_priority was used
    model_for_av = CCM89(Rv=3.1)  # rv in deredden_flux is 3.1 by default
    atten_av = model_for_av.extinguish(
        wavelengths,
        Av=av_priority.value if isinstance(av_priority, u.Quantity) else av_priority,
    )
    expected_flux_av = flux / atten_av

    dered_flux_with_both = deredden_flux(
        wavelengths,
        flux,
        model_class=CCM89,
        av=av_priority,
        ebv=ebv_ignored,
        rv=rv_ignored,
    )
    assert_allclose(dered_flux_with_both.value, expected_flux_av.value, rtol=1e-6)


# Tests with Dummy Models to check internal logic paths


def test_dummy_rv_model_ebv(basic_inputs_units):
    """Test deredden_flux with DummyRvModel using E(B-V)."""
    wavelengths, flux, params_ebv, _ = basic_inputs_units
    rv = params_ebv["rv"]
    ebv = params_ebv["ebv"]

    dered_flux_q = deredden_flux(
        wavelengths, flux, model_class=DummyRvModel, ebv=ebv, rv=rv
    )

    # DummyRvModel: A(x)/A(V) = x / Rv_instance
    # model_instance.extinguish(Ebv=ebv) uses model_instance.Rv
    # A(V)_calc = model_instance.Rv * ebv
    # A_lambda = (x_inv_micron / model_instance.Rv) * A(V)_calc
    #          = (x_inv_micron / rv) * (rv * ebv) = x_inv_micron * ebv
    x_inv_micron = 1.0 / wavelengths.to(u.micron).value
    alambda_manual = x_inv_micron * ebv
    attenuation_manual = 10 ** (-0.4 * alambda_manual)
    expected_dered_flux_values = flux.value / attenuation_manual

    assert_allclose(dered_flux_q.value, expected_dered_flux_values, rtol=1e-5)


def test_dummy_non_rv_model_av(basic_inputs_units):
    """Test deredden_flux with DummyNonRvModel using A(V)."""
    wavelengths, flux, _, params_av = basic_inputs_units
    av = params_av["av"]

    dered_flux_q = deredden_flux(wavelengths, flux, model_class=DummyNonRvModel, av=av)

    # DummyNonRvModel: A(x)/A(V) = x
    # A_lambda = x_inv_micron * Av
    x_inv_micron = 1.0 / wavelengths.to(u.micron).value
    alambda_manual = x_inv_micron * (av.value if isinstance(av, u.Quantity) else av)
    attenuation_manual = 10 ** (-0.4 * alambda_manual)
    expected_dered_flux_values = flux.value / attenuation_manual

    assert_allclose(dered_flux_q.value, expected_dered_flux_values, rtol=1e-5)


def test_dummy_non_rv_model_ebv_error(basic_inputs_units):
    """Test error using E(B-V) with DummyNonRvModel."""
    wavelengths, flux, params_ebv, _ = basic_inputs_units
    with pytest.raises(
        ValueError,
        match="Model DummyNonRvModel is not R(V)-dependent. A(V) must be provided",
    ):
        deredden_flux(
            wavelengths,
            flux,
            model_class=DummyNonRvModel,
            ebv=params_ebv["ebv"],
            rv=params_ebv["rv"],
        )
