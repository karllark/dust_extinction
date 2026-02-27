import numpy as np
import pytest

from dust_extinction.conv_functions import unred
from dust_extinction.parameter_averages import G23, F99


class TestUnred:
    """Test unred functionality"""

    def test_basic_functionality(self):
        """Test basic functionality of unred function"""
        # Create test data
        wave = np.linspace(3000, 8000, 100)  # 3000-8000 Angstroms
        flux = np.random.random(100)  # Random flux values
        ebv = 0.1

        # Test unred function (default should be G23)
        dereddened_flux = unred(wave, flux, ebv)
        assert dereddened_flux.shape == flux.shape
        assert not np.array_equal(dereddened_flux, flux)  # Should be different

    def test_reddening(self):
        """Test that negative E(B-V) reddens spectrum"""
        wave = np.linspace(3000, 8000, 100)
        flux = np.ones(100)  # Flat flux
        ebv_positive = 0.1
        ebv_negative = -0.1

        # Positive E(B-V) should deredden (increase flux)
        dereddened = unred(wave, flux, ebv_positive)

        # Negative E(B-V) should redden (decrease flux)
        reddened = unred(wave, flux, ebv_negative)

        # Check that dereddened flux > original > reddened flux
        assert np.mean(dereddened) > np.mean(flux)
        assert np.mean(flux) > np.mean(reddened)

    def test_rv_dependence(self):
        """Test that different R_V values produce different results"""
        wave = np.linspace(3000, 8000, 100)
        flux = np.random.random(100)
        ebv = 0.1

        # Test with different R_V values (G23 valid range: 2.3-5.6)
        result_rv2 = unred(wave, flux, ebv, R_V=2.5)
        result_rv3 = unred(wave, flux, ebv, R_V=3.1)
        result_rv4 = unred(wave, flux, ebv, R_V=4.0)

        # Results should be different
        assert not np.array_equal(result_rv2, result_rv3)
        assert not np.array_equal(result_rv3, result_rv4)
        assert not np.array_equal(result_rv2, result_rv4)

    def test_g23_vs_f99(self):
        """Test that G23 (default) and F99 give different results"""
        wave = np.linspace(3000, 8000, 100)
        flux = np.random.random(100)
        ebv = 0.1

        # Default should be G23
        result_g23 = unred(wave, flux, ebv)

        # Test with explicit F99
        f99_model = F99(Rv=3.1)
        result_f99 = unred(wave, flux, ebv, ext_model=f99_model)

        # Should be different (though close in optical range)
        assert not np.array_equal(result_g23, result_f99)

    def test_error_handling(self):
        """Test error handling"""
        wave = np.linspace(3000, 8000, 100)
        flux = np.random.random(50)  # Different size
        ebv = 0.1

        # Should raise an error
        with pytest.raises(ValueError, match="wave and flux must have the same shape"):
            unred(wave, flux, ebv)

    def test_single_wavelength(self):
        """Test with single wavelength values"""
        wave = np.array([5000.0])  # Single wavelength
        flux = np.array([1.0])  # Single flux
        ebv = 0.1

        # Should work with scalar values
        result = unred(wave, flux, ebv)
        assert result.shape == flux.shape
        assert result[0] > flux[0]  # Should be dereddened

    def test_wavelength_range(self):
        """Test at valid wavelength range extremes"""
        # Test at valid range limits
        wave_short = np.array([1000.0])  # Near UV limit
        wave_long = np.array([10000.0])  # Near IR limit
        flux = np.array([1.0])
        ebv = 0.1

        # These should work without errors
        result_short = unred(wave_short, flux, ebv)
        result_long = unred(wave_long, flux, ebv)

        assert result_short.shape == flux.shape
        assert result_long.shape == flux.shape

    def test_units_handling(self):
        """Test that function handles different input units properly"""
        # Test with simple arrays (no units)
        wave = np.linspace(3000, 8000, 100)
        flux = np.random.random(100)
        ebv = 0.1

        # Should work with plain numpy arrays
        result = unred(wave, flux, ebv)
        assert result.shape == flux.shape

    def test_extinction_model_parameter(self):
        """Test using custom extinction models"""
        wave = np.linspace(3000, 8000, 100)
        flux = np.random.random(100)
        ebv = 0.1

        # Test with G23 model directly
        g23_model = G23(Rv=3.0)
        result_g23_custom = unred(wave, flux, ebv, ext_model=g23_model)

        # Test with F99 model directly
        f99_model = F99(Rv=4.0)
        result_f99_custom = unred(wave, flux, ebv, ext_model=f99_model)

        # Results should be different
        assert not np.array_equal(result_g23_custom, result_f99_custom)

    def test_zero_ebv(self):
        """Test that zero E(B-V) returns original flux"""
        wave = np.linspace(3000, 8000, 100)
        flux = np.random.random(100)
        ebv = 0.0

        result = unred(wave, flux, ebv)
        np.testing.assert_array_almost_equal(result, flux)

    def test_consistency_check(self):
        """Test that unred + redden gives back original flux"""
        wave = np.linspace(3000, 8000, 100)
        original_flux = np.random.random(100)
        ebv = 0.1

        # Deredden then redden should give back original
        dereddened = unred(wave, original_flux, ebv)
        recovered = unred(wave, dereddened, -ebv)

        np.testing.assert_array_almost_equal(original_flux, recovered, decimal=10)

    def test_default_is_g23(self):
        """Test that default unred uses G23 model"""
        wave = np.linspace(3000, 8000, 100)
        flux = np.random.random(100)
        ebv = 0.1

        # Default should use G23
        result_default = unred(wave, flux, ebv)

        # Compare with explicit G23
        g23_model = G23(Rv=3.1)
        result_g23_explicit = unred(wave, flux, ebv, ext_model=g23_model)

        np.testing.assert_array_almost_equal(result_default, result_g23_explicit)
