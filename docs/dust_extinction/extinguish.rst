Extinguishing Flux (Applying Reddening)
=======================================

The ``extinguish`` method is used to extinguish flux by dust extinction.
The ``extinguish`` method supports ``astropy.units`` (the ``flux`` parameter).
If ``flux`` is an Astropy ``Quantity``, then the returned ``Quantity`` will
have the same units. If ``flux`` is a ``float`` or ``numpy.ndarray``, then the
returned value will also be a ``float`` or ``numpy.ndarray``.  The ``wavelength``
input must be in `1/micron` or be an Astropy ``Quantity`` with spectral equivalence.
The ``Av`` or ``Ebv`` inputs are floats.

Example:
~~~~~~~~

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   import astropy.units as u
   from astropy.visualization import quantity_support
   quantity_support()

   from dust_extinction.parameter_averages import CCM89
   from dust_extinction.helpers importExtCurve

   # generate the curves and plot them
   fig, ax = plt.subplots()

   # temp model to get the correct x range
   text_model =ExtCurve()

   # generate the curves and plot them
   x = np.arange(text_model.x_range[0], text_model.x_range[1],0.1)/u.micron

   # define a flat continuum source
   source_flux = np.ones(len(x))

   # define an extinction model
   ext_model = CCM89(Rv=3.1)

   # extinguish (redden) the source flux
   #   the Av parameter is the full extinction in V
   #   internally, the model calculates Alambda/Av and then multiplies by Av
   #   the result is Alambda
   #   the user just needs to specify Av
   #   the model is defined in terms of Alambda/Av, the shape of the curve
   #   not the full amount of extinction
   #   the amount of extinction is set by Av
   #   extinguish returns 10**(-0.4*Alambda)
   #   Alambda is the extinction in magnitudes at each wavelength
   #   the value returned by extinguish is the scaling factor for the flux
   #   flux_red = flux_orig*10**(-0.4*Alambda)
   #   flux_red = flux_orig*ext_model.extinguish(x, Av=1.0)
   flux_ext = source_flux*ext_model.extinguish(x, Av=1.0)

   ax.plot(x, source_flux, label='source')
   ax.plot(x, flux_ext, label='extinguished')

   ax.set_xlabel('$x$ [$\mu m^{-1}$]')
   ax.set_ylabel('Flux')
   ax.set_title('Extinction with CCM89 (Rv=3.1)')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

This is a simple example that uses the CCM89 model.  Other models are available
(see :ref:`models_table`).

More examples are available in an `Astropy tutorial
<http://learn.astropy.org/rst-tutorials/Dust-Extinction-Absorption-Correcting-Flux.html?highlight=extinction>`_
and a `Jupyter notebook <https://github.com/karllark/dust_extinction/blob/master/examples/plot_extinction.ipynb>`_
that was created as part of the
`Learn.Astropy
<https://learn.astropy.org/>`_ effort.


Dereddening Flux with ``deredden_flux``
=======================================

While dereddening can be achieved by dividing an observed flux by the
output of the ``model.extinguish()`` method (as ``extinguish()`` returns
the attenuation factor :math:`10^{-0.4 A_\lambda}`), the package also
provides a more generalized convenience function ``deredden_flux`` for this task.

This function allows you to use various extinction models from the
``dust_extinction`` package. You can specify the amount of extinction
either by ``ebv`` (color excess E(B-V), typically used with R(V)-dependent
models like ``CCM89``, ``F99``, etc.) or directly by ``av`` (total V-band
extinction A(V), which can be used with any model).

Key parameters for ``deredden_flux``:
  - ``wavelengths``: The wavelength data.
  - ``flux``: The observed flux data.
  - ``model_class``: The extinction model class to use (e.g., ``CCM89``, ``F99``, ``G03_SMCBar``).
  - ``av``: A(V) value. Takes precedence if provided.
  - ``ebv``: E(B-V) value. Used if ``av`` is not provided.
  - ``rv``: R(V) value (default 3.1). Used with ``ebv`` if the model is R(V)-dependent.

Example using an R(V)-dependent model (CCM89) with E(B-V):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import numpy as np

   import astropy.units as u
   from astropy.modeling.models import BlackBody

   from dust_extinction.parameter_averages import CCM89 # For reddening simulation
   from dust_extinction import deredden_flux # The generalized function

   # Generate wavelengths
   lam_obs = np.array([0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.6, 2.2]) * u.micron

   # Create a synthetic intrinsic spectrum
   bb_intrinsic = BlackBody(temperature=5000*u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))
   flux_intrinsic = bb_intrinsic(lam_obs)

   # Define extinction parameters for CCM89
   ebv_val = 0.3
   rv_val = 3.1

   # Redden this intrinsic flux to simulate an observed flux
   ccm_model_sim = CCM89(Rv=rv_val)
   attenuation_factor = ccm_model_sim.extinguish(lam_obs, Ebv=ebv_val) # Use .value for Ebv if it has units
   flux_observed = flux_intrinsic * attenuation_factor

   # Now, deredden flux_observed using deredden_flux with CCM89
   flux_dereddened_ebv = deredden_flux(lam_obs, flux_observed,
                                       model_class=CCM89,
                                       ebv=ebv_val, rv=rv_val)

   # Plot the results
   fig, ax = plt.subplots()

   ax.plot(lam_obs, flux_intrinsic, label='Intrinsic Flux', marker='o', linestyle='--')
   ax.plot(lam_obs, flux_observed, label=f'Observed Flux (CCM89, E(B-V)={ebv_val}, Rv={rv_val})', marker='x')
   ax.plot(lam_obs, flux_dereddened_ebv, label='Dereddened Flux (deredden_flux with CCM89, E(B-V))', marker='s', linestyle=':')

   ax.set_xlabel(f"Wavelength [{lam_obs.unit}]")
   ax.set_ylabel(f"Flux [{flux_intrinsic.unit}]")
   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())
   ax.set_yscale('log')
   ax.yaxis.set_major_formatter(ScalarFormatter())
   ax.set_title('Dereddening with deredden_flux (R(V)-dependent model)')
   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

Example using a non-R(V)-dependent model (G03_SMCBar) with A(V):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If using a model that is not R(V)-dependent, or if you know A(V) directly,
you should use the ``av`` parameter.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import numpy as np

   import astropy.units as u
   from astropy.modeling.models import BlackBody

   from dust_extinction.averages import G03_SMCBar # A non-R(V) model for reddening
   from dust_extinction import deredden_flux

   # Re-use lam_obs, bb_intrinsic, flux_intrinsic from previous example
   lam_obs = np.array([0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.6, 2.2]) * u.micron
   bb_intrinsic = BlackBody(temperature=5000*u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))
   flux_intrinsic = bb_intrinsic(lam_obs)

   # Define extinction parameters for G03_SMCBar
   av_val_smc = 0.9 # Example A(V)

   # Redden flux using G03_SMCBar
   smc_model_sim = G03_SMCBar()
   attenuation_smc = smc_model_sim.extinguish(lam_obs, Av=av_val_smc) # Use .value for Av if it has units
   flux_observed_smc = flux_intrinsic * attenuation_smc

   # Deredden using deredden_flux with G03_SMCBar and av
   flux_dereddened_av = deredden_flux(lam_obs, flux_observed_smc,
                                      model_class=G03_SMCBar,
                                      av=av_val_smc)

   # Plot
   fig, ax = plt.subplots()
   ax.plot(lam_obs, flux_intrinsic, label='Intrinsic Flux', marker='o', linestyle='--')
   ax.plot(lam_obs, flux_observed_smc, label=f'Observed Flux (G03_SMCBar, A(V)={av_val_smc})', marker='x')
   ax.plot(lam_obs, flux_dereddened_av, label='Dereddened Flux (deredden_flux with G03_SMCBar, A(V))', marker='s', linestyle=':')
   ax.set_xlabel(f"Wavelength [{lam_obs.unit}]")
   ax.set_ylabel(f"Flux [{flux_intrinsic.unit}]")
   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())
   ax.set_yscale('log')
   ax.yaxis.set_major_formatter(ScalarFormatter())
   ax.set_title('Dereddening with deredden_flux (Non-R(V) model)')
   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

These examples demonstrate how ``deredden_flux`` can be used with different
types of models and extinction parameters. The dereddened flux should ideally
recover the intrinsic flux, apart from minor numerical precision differences.
