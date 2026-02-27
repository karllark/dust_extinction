import numpy as np
import astropy.units as u

from . import parameter_averages

__all__ = ["unred"]


def unred(wave, flux, ebv, ext_model=None, R_V=3.1):
    """
    Deredden or redden a flux vector using a specified extinction model.

    This is a general function that can work with any extinction model
    that follows the dust_extinction interface.

    Parameters
    ----------
    wave : array_like
        Wavelength vector in Angstroms
    flux : array_like
        Calibrated flux vector, same number of elements as wave
    ebv : float
        Color excess E(B-V), scalar. If a negative EBV is supplied,
        then fluxes will be reddened rather than dereddened.
    ext_model : extinction model, optional
        Extinction model instance (e.g., G23(Rv=R_V), F99(Rv=R_V))
        If not specified, defaults to G23
    R_V : float, optional
        Ratio of total to selective extinction, A(V)/E(B-V)
        Default is 3.1. Ignored if ext_model is provided.

    Returns
    -------
    flux_corrected : ndarray
        Dereddened flux vector, same units and number of elements as flux

    Raises
    ------
    ValueError
        If wave and flux arrays have different sizes

    Notes
    -----
    Based on IDL astrolib routine CCM_UNRED, but using modern G23 extinction.

    The correction applied is:
    F_corrected = F_observed * 10^(0.4 * A(λ) * E(B-V) * R_V)

    where A(λ) is calculated from the specified extinction model.

    Examples
    --------
    >>> import numpy as np
    >>> from dust_extinction.unred import unred
    >>>
    >>> # Example wavelengths (3000-8000 Angstroms)
    >>> wave = np.linspace(3000, 8000, 100)
    >>> flux = np.random.random(100)  # Some mock flux values
    >>> ebv = 0.1
    >>>
    >>> # General usage with default G23 model
    >>> dereddened_flux = unred(wave, flux, ebv)
    >>>
    >>> # Using explicit model
    >>> from dust_extinction.parameter_averages import F99
    >>> f99_model = F99(Rv=3.1)
    >>> dereddened_flux = unred(wave, flux, ebv, ext_model=f99_model)
    """
    # Convert inputs to numpy arrays
    wave = np.asarray(wave)
    flux = np.asarray(flux)

    # Check input consistency
    if wave.shape != flux.shape:
        raise ValueError("wave and flux must have the same shape")

    # Set default extinction model if not provided
    if ext_model is None:
        ext_model = parameter_averages.G23(Rv=R_V)

    # Convert wavelength from Angstroms to inverse microns for extinction model
    # 1 Angstrom = 1e-4 microns, so 1/Angstrom = 1e4 1/microns
    x_wave = (1e4 / wave) / u.micron

    # Calculate extinction curve A(λ)/A(V)
    a_lambda_over_av = ext_model(x_wave)

    # Calculate the correction factor
    # For dereddening: multiply by 10^(0.4 * A(λ) * E(B-V) * R_V)
    # For reddening (negative ebv): same formula works
    correction_factor = np.power(10.0, 0.4 * a_lambda_over_av * ebv * R_V)

    # Apply correction
    return flux * correction_factor
