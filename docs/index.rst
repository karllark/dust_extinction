.. _dust-extinction-mainpage:

############################
Interstellar Dust Extinction
############################

``dust_extinction`` is a python package to provide models of interstellar dust
extinction curves.

Extinction describes the effects of dust on observations of single star due to
the dust along the line-of-sight to a star removing flux by absorbing photons
and scattering photons out of the line-of-sight. The wavelength dependence of
dust extinction (also know as extinction curves) provides fundamental
information about the size, composition, and shape of interstellar dust grain.
In general, extinction models are used to model or correct the effects of dust
on observations.

In contrast, dust attenuation refers to the effects of dust on the measurements
of groups of stars mixed with dust or stars with circumstellar dust.
See :ref:`ExtvsAtt`.  For attenuation models, see the `dust_attenuation
package <https://dust-attenuation.readthedocs.io/>`_.

This package is an
`astropy affiliated package <https://www.astropy.org/affiliated/>`_
and uses the
`astropy.modeling <https://docs.astropy.org/en/stable/modeling/>`_
framework.

Citation
========

Please cite `Gordon 2024, JOSS, 9(100), 7023 <https://joss.theoj.org/papers/10.21105/joss.07023>`_.

In additon, please cite the original papers for the models used.  The papers for each model 
are given `here <https://dust-extinction.readthedocs.io/en/latest/dust_extinction/references.html>`_.

User Documentation
==================

.. toctree::
   :maxdepth: 2

   Extinction versus Attenuation (what is the difference) <dust_extinction/extinction.rst>
   Flavors of Models (including plots of all models) <dust_extinction/model_flavors.rst>
   How to choose a model (including listing of all models) <dust_extinction/choose_model.rst>
   Extinguish (or unextinguish) data (how to correct/account for extinction) <dust_extinction/extinguish.rst>
   Fitting extinction curves <dust_extinction/fit_extinction.rst>
   References <dust_extinction/references.rst>

Dev Documentation
=================

.. toctree::
   :maxdepth: 1

   Model Base Classes (how to add a model) <dust_extinction/dev_model.rst>

Installation
============

.. toctree::
  :maxdepth: 2

  How to install <dust_extinction/install.rst>

Repository
==========

GitHub: `dust_extinction <https://github.com/karllark/dust_extinction>`_

Quick Start
===========

Extinction Curve
----------------

How to get the A(x)/A(V) extinction curve for an input set of x wavelength values.

Define a model, specifically the G23 model with an R(V) = 3.1.

.. code-block:: python

    from dust_extinction.parameter_averages import G23

    # define the model
    extmod = G23(Rv=3.1)

Define the wavelengths 

.. code-block:: python

    wavelengths = np.logspace(np.log10(0.1), np.log10(30.0), num=1000)*u.micron

Get the extinction values in A(lambda)/A(V) units.

.. code-block:: python

    ext = extmod(wavelengths)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from dust_extinction.parameter_averages import G23

    # define the model
    extmod = G23(Rv=3.1)

    # wavelengths as 1D arrays
    wavelengths = np.logspace(np.log10(0.1), np.log10(30.0), num=1000)*u.micron

    # extinction at the wavelengths
    ext = extmod(wavelengths)

    # plot the intrinsic and extinguished fluxes
    fig, ax = plt.subplots()

    ax.plot(wavelengths, ext, label='G23 (Rv=3.1)', linewidth=6, alpha=0.5)

    ax.set_xlabel(r'$\lambda$ [{}]'.format(wavelengths.unit))
    ax.set_ylabel(r'$A(\lambda)/A(V)$')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

Extinguish/Unextinguish a Spectrum
----------------------------------

How to extinguish (redden) and unextinguish (deredden) a spectrum:

Generate a spectrum to use.  In this case a blackbody model, but can be an
observed spectrum.  The `dust_extinction` models are unit aware and the
wavelength array should have astropy.units associated with it.

.. code-block:: python

    import numpy as np
    from astropy.modeling.models import BlackBody
    import astropy.units as u

    # wavelengths and spectrum are 1D arrays
    # wavelengths between 1000 and 30000 A
    wavelengths = np.logspace(np.log10(0.1), np.log10(30.0), num=1000)*u.micron
    bb_lam = BlackBody(10000*u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
    spectrum = bb_lam(wavelengths)

Define a model, specifically the G23 model with an R(V) = 3.1.

.. code-block:: python

    from dust_extinction.parameter_averages import G23

    # define the model
    ext = G23(Rv=3.1)

Extinguish (redden) a spectrum with a screen of G23 dust with an E(B-V) of 0.5.
Can also specify the dust column with Av (this case equivalent to Av = 0.5*Rv =
1.55).

.. code-block:: python

    # extinguish (redden) the spectrum
    spectrum_ext = spectrum*ext.extinguish(wavelengths, Ebv=0.5)

Unextinguish (deredden) a spectrum with a screen of G23 dust with the
equivalent A(V) column.

.. code-block:: python

    # unextinguish (deredden) the spectrum
    # Av = 1.55 = R(V) * E(B-V) = 3.1 * 0.5
    spectrum_noext = spectrum_ext/ext.extinguish(wavelengths, Av=1.55)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.modeling.models import BlackBody
    from dust_extinction.parameter_averages import G23

    # define the model
    ext = G23(Rv=3.1)

    # wavelengths and spectrum are 1D arrays
    wavelengths = np.logspace(np.log10(0.1), np.log10(30.0), num=1000)*u.micron
    bb_lam = BlackBody(10000*u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
    spectrum = bb_lam(wavelengths)

    # extinguish (redden) the spectrum
    spectrum_ext = spectrum*ext.extinguish(wavelengths, Ebv=0.5)

    # unextinguish (deredden) the spectrum
    # Av = 1.55 = R(V) * E(B-V) = 3.1 * 0.5
    spectrum_noext = spectrum_ext/ext.extinguish(wavelengths, Av=1.55)

    # plot the intrinsic and extinguished fluxes
    fig, ax = plt.subplots()

    ax.plot(wavelengths, spectrum, label='spectrum', linewidth=6, alpha=0.5)
    ax.plot(wavelengths, spectrum_ext, label='spectrum_ext')
    ax.plot(wavelengths, spectrum_noext, 'k', label='spectrum_noext')

    ax.set_xlabel('$\lambda$ [{}]'.format(wavelengths.unit))
    ax.set_ylabel('$Flux$ [{}]'.format(spectrum.unit))

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

Reporting Issues
================

If you have found a bug in ``dust_extinction`` please report it by creating a
new issue on the ``dust_extinction`` `GitHub issue tracker
<https://github.com/karllark/dust_extinction/issues>`_.

Please include an example that demonstrates the issue sufficiently so that the
developers can reproduce and fix the problem. You may also be asked to provide
information about your operating system and a full Python stack trace.  The
developers will walk you through obtaining a stack trace if it is necessary.

Contributing
============

Like the `Astropy`_ project, ``dust_extinction`` is made both by and for its
users.  We accept contributions at all levels, spanning the gamut from fixing a
typo in the documentation to developing a major new feature. We welcome
contributors who will abide by the `Python Software Foundation Code of Conduct
<https://www.python.org/psf/conduct/>`_.

``dust_extinction`` follows the same workflow and coding guidelines as
`Astropy <https://docs.astropy.org/>`_.  Take a look at the astropy
`developer <https://docs.astropy.org/en/latest/index_dev.html>`_ documentation for
guidelines.

For the complete list of contributors please see the `dust_extinction
contributors page on Github
<https://github.com/karllark/dust_extinction/graphs/contributors>`_.

Reference API
=============

.. toctree::
   :maxdepth: 2

   dust_extinction/reference_api.rst
