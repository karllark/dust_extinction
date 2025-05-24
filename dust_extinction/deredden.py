import numpy as np
from astropy import units as u

from .parameter_averages import CCM89 # Still default model
from .baseclasses import BaseExtModel, BaseExtRvModel

def deredden_flux(wavelengths, flux, model_class, av=None, ebv=None, rv=3.1):
    """
    Deredden a flux array using a specified dust extinction model.

    This function generalizes the concept of dereddening to work with
    various models available in the dust_extinction package.

    Parameters
    ----------
    wavelengths : array_like or astropy.units.Quantity
        Wavelength values. If not a Quantity, assumed to be in microns
        unless the chosen model has different default expectations.
        The validity range depends on the chosen model.
    flux : array_like or astropy.units.Quantity
        Observed flux values. If a Quantity, its units are preserved.
    model_class : dust_extinction.baseclasses.BaseExtModel subclass
        The extinction model class to use (e.g., CCM89, F99, O94 for
        R(V)-dependent; or other models if they support an `extinguish`
        method compatible with Av input).
    av : float, optional
        Total extinction A(V) in magnitudes. If provided, `av` takes
        precedence over `ebv` and `rv`.
    ebv : float, optional
        Color excess E(B-V) in magnitudes. Used if `av` is not provided.
        Requires the model to be R(V)-dependent (e.g., subclass of
        BaseExtRvModel or initialized with `rv` if applicable) or `rv`
        to be explicitly passed.
    rv : float, optional
        Ratio of total to selective extinction A(V)/E(B-V). Default is 3.1.
        Used when `ebv` is provided and the model is R(V)-dependent.
        Ignored if `av` is provided.

    Returns
    -------
    flux_dereddened : ndarray or astropy.units.Quantity
        Dereddened flux values, of the same type and units (if any) as `flux`.

    Raises
    ------
    TypeError
        If `model_class` is not a subclass of BaseExtModel.
    ValueError
        If input parameters are insufficient (e.g., neither `av` nor `ebv`
        provided), or if `ebv` is used with a model that cannot determine
        `Av` from it (e.g., not R(V)-dependent and `rv` not useful).
        Also, if `ebv` or `av` is negative.

    Notes
    -----
    The core calculation is `flux_dereddened = flux / attenuation_factor`,
    where `attenuation_factor` is obtained from `model_instance.extinguish(...)`.

    If `av` is given, it's used directly.
    If `ebv` is given and `av` is not:
        - If `model_class` is a subclass of `BaseExtRvModel` (like CCM89, O94, F99),
          it's initialized with `Rv=rv`, and then `extinguish` is called with `Ebv=ebv`.
        - If `model_class` is not R(V)-dependent, using `ebv` is an error because
          `Av` cannot be derived without `Rv`. In such cases, `av` must be supplied.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from dust_extinction.parameter_averages import CCM89, F99
    >>> # Deredden using CCM89 (R(V)-dependent) with E(B-V)
    >>> wave = np.array([0.4, 0.5, 0.6]) * u.micron
    >>> obs_flux = np.array([1.0, 1.5, 2.0]) * u.Jy
    >>> ebv_val = 0.2
    >>> rv_val = 3.1
    >>> dered_flux_ebv = deredden_flux(wave, obs_flux, model_class=CCM89,
    ...                                ebv=ebv_val, rv=rv_val)
    >>> print(dered_flux_ebv) # doctest: +SKIP
    [ 2.30786196  2.84713805  3.35656508] Jy

    >>> # Deredden using F99 (another R(V)-dependent model) with A(V)
    >>> av_val = 0.62 # Equivalent to Ebv=0.2, Rv=3.1
    >>> dered_flux_av = deredden_flux(wave, obs_flux, model_class=F99, av=av_val)
    >>> print(dered_flux_av) # doctest: +SKIP
    # Values will differ from CCM89 due to different model prescription
    [ 2.23677549  2.79653466  3.33099008] Jy
    """
    if not issubclass(model_class, BaseExtModel):
        raise TypeError("model_class must be a subclass of BaseExtModel.")

    if not issubclass(model_class, BaseExtModel):
        raise TypeError("model_class must be a subclass of BaseExtModel.")

    # Strip units from av and ebv if they are Quantities, model.extinguish expects float
    av_value = av.value if isinstance(av, u.Quantity) else av
    ebv_value = ebv.value if isinstance(ebv, u.Quantity) else ebv
    # rv is already expected to be float

    if av_value is None and ebv_value is None:
        raise ValueError("Either 'av' or 'ebv' must be provided.")

    # Validate av and ebv values (use the unit-stripped values for check)
    if av_value is not None and av_value < 0.0:
        raise ValueError(f"A(V) = {av_value} must be non-negative.")
    # Check ebv_value only if av_value is not provided (av takes precedence)
    if av_value is None and ebv_value is not None and ebv_value < 0.0:
        raise ValueError(f"E(B-V) = {ebv_value} must be non-negative.")

    model_instance = None
    attenuation_factor = None

    # Ensure wavelengths are in appropriate units for the model
    if not isinstance(wavelengths, u.Quantity):
        waves_for_model = wavelengths * u.micron # Default assumption
    else:
        waves_for_model = wavelengths

    if av_value is not None: # A(V) takes precedence
        # For models that might still need Rv for their own setup, even if Av is given to extinguish
        if issubclass(model_class, BaseExtRvModel):
            model_instance = model_class(Rv=rv)
        else:
            try:
                model_instance = model_class()
            except Exception as e: # pragma: no cover
                 raise RuntimeError(
                    f"Failed to initialize model {model_class.__name__} without parameters. "
                    f"Original error: {e}"
                )
        attenuation_factor = model_instance.extinguish(waves_for_model, Av=av_value)
    elif ebv_value is not None: # Use E(B-V)
        if issubclass(model_class, BaseExtRvModel):
            model_instance = model_class(Rv=rv)
            # This extinguish call will use model_instance.Rv internally
            attenuation_factor = model_instance.extinguish(waves_for_model, Ebv=ebv_value)
        else:
            # Non-R(V) dependent models cannot use E(B-V) to determine A(V)
            # as they don't have an Rv attribute intrinsically.
            # The base BaseExtModel.extinguish(Ebv=...) relies on self.Rv.
            raise ValueError(
                f"Model {model_class.__name__} is not R(V)-dependent. "
                f"A(V) must be provided directly via the 'av' parameter "
                f"when using this model."
            )
    else: # Should be caught by the initial check, but as a safeguard
        raise ValueError("Internal error: No valid extinction parameter determined.") # pragma: no cover

    flux_dereddened = flux / attenuation_factor
    return flux_dereddened
