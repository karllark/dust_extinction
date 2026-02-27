.. _extinguish_example:

###############################
Extinguish or Unextinguish Data
###############################

Two of the three flavors of models include a function to calculate the
factor to multiple (extinguish) or divide (unextinguish) a spectrum by
to add or remove the effects of dust, respectively.

Extinguish is also often called reddening.  Extinguishing a spectrum often
reddens the flux, but sometimes 'bluens' the flux
(e.g, on the short wavelength side of the 2175 A bump).
So extinguish is the more generic term.

Extinguish a Blackbody
======================

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import numpy as np

   import astropy.units as u
   from astropy.modeling.models import BlackBody

   from dust_extinction.parameter_averages import G23

   # generate wavelengths between 0.092 and 31 microns
   #    within the valid range for the G23 R(V) dependent relationship
   lam = np.logspace(np.log10(0.092), np.log10(31.0), num=1000)

# setup the inputs for the blackbody function
    wavelengths = lam*1e4*u.AA
    
    # get the blackbody flux
    bb_lam = BlackBody(10000*u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
    flux = bb_lam(wavelengths)
    
    # initialize the model
    ext = G23(Rv=3.1)
    
    # plot the intrinsic, extinguished, and unextinguished flux
    fig, ax = plt.subplots()
    ax.plot(wavelengths, flux, label='Intrinsic')
    ax.plot(wavelengths, flux*ext.extinguish(wavelengths, Av=0.5), label='$A(V) = 0.5$')
    ax.plot(wavelengths, flux*ext.extinguish(wavelengths, Av=1.5), label='$A(V) = 1.5$')
    ax.plot(wavelengths, flux*ext.extinguish(wavelengths, Ebv=1.0), label='$E(B-V) = 1.0$')
    ax.set_xlabel(r'$\lambda$ [$\mu$m]')
    ax.set_ylabel(r'$Flux$')
    ax.set_title('Extinguish/Unextinguish a Blackbody')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yscale('log')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # Note: The G23 model is used for extinction instead of a custom extinguish model
    # For custom extinguish models, see the grain_models module
    
Unreddening/Reddening
=======================

The `dust_extinction.conv_functions.unred` module provides a general function to
deredden or redden spectra using any dust extinction model. This is a modern
implementation of the classic IDL `ccm_unred` function, but uses the Gordon et
al. (2023) extinction model by default.

.. autofunction:: dust_extinction.conv_functions.unred

.. py:function:: unred
    .. py:function:: unred(wave, flux, ebv, ext_model=None, R_V=3.1)

   Deredden or redden a flux vector using a specified extinction model.

   This is a general function that can work with any extinction model
   that follows the dust_extinction interface. If no model is provided,
   the modern Gordon et al. (2023) extinction model (G23) is used by default.

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
       If not specified, defaults to G23.
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
   >>> flux = np.random.random(100)
   >>> ebv = 0.1
   >>> 
   >>> # General usage with default G23 model
   >>> dereddened_flux = unred(wave, flux, ebv)
   >>> 
   >>> # Reddening (use negative ebv)
   >>> reddened_flux = unred(wave, flux, -ebv)
   >>> 
   >>> # Using alternative models
   >>> from dust_extinction.parameter_averages import F99
   >>> f99_model = F99(Rv=3.1)
   >>> dereddened_f99 = unred(wave, flux, ebv, ext_model=f99_model)

.. note::
   The default G23 model is based on the most up-to-date extinction science
   and provides a modern replacement for the classic IDL CCM89 function.
   
   Users who need specific alternative extinction laws (like F99) can still access them
   via the ext_model parameter, while most users will prefer the modern G23 default.

   # initialize the model
   ext = G23(Rv=3.1)

   # get the extinguished blackbody flux for different amounts of dust
   flux_ext_av05 = flux*ext.extinguish(wavelengths, Av=0.5)
   flux_ext_av15 = flux*ext.extinguish(wavelengths, Av=1.5)
   flux_ext_ebv10 = flux*ext.extinguish(wavelengths, Ebv=1.0)

   # plot the intrinsic and extinguished fluxes
   fig, ax = plt.subplots()

   ax.plot(wavelengths, flux, label='Intrinsic')
   ax.plot(wavelengths, flux_ext_av05, label='$A(V) = 0.5$')
   ax.plot(wavelengths, flux_ext_av15, label='$A(V) = 1.5$')
   ax.plot(wavelengths, flux_ext_ebv10, label='$E(B-V) = 1.0$')

   ax.set_xlabel('$\lambda$ [$\AA$]')
   ax.set_ylabel('$Flux$')

   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())
   ax.set_yscale('log')

   ax.set_title('Example extinguishing a blackbody')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

Notebooks
=========

A great way to show how to use the `dust_extinction` package is using a
jupyter notebook.  Check out the
`Analyzing interstellar reddening and calculating synthetic photometry
<https://learn.astropy.org/tutorials/color-excess.html>`_
notebook that was created as part of the
`Learn.Astropy
<https://learn.astropy.org/>`_ effort.
