.. _dust-extinction-mainpage:

############################
Interstellar Dust Extinction
############################

``dust_extinction`` is a python package to provide models of interstellar dust
extinction curves.

Extinction describes the effects of dust on a single star.  The dust along the
line-of-sight to the stars removes flux by absorbing photons or scattering
photons out of the line-of-sight.  In general, extinction models are used to
model or correct the effects of dust on observations a single star.

In contrast, dust attenuation refers to the effects of dust on the measurements
of groups of stars mixed with dust or stars with circumstellar dust.
See :ref:`ExtvsAtt`.  For attenuation models, see the `dust_attenuation
package <https://dust-attenuation.readthedocs.io/>`_.

This package is an
`astropy affiliated package <https://www.astropy.org/affiliated/>`_
and uses the
`astropy.modeling <https://docs.astropy.org/en/stable/modeling/>`_
framework.

User Documentation
==================

.. toctree::
   :maxdepth: 2

   Extinction versus Attenuation <dust_extinction/extinction.rst>
   Flavors of Models <dust_extinction/model_flavors.rst>
   How to choose a model <dust_extinction/choose_model.rst>
   Extinguish (or unextinguish) data <dust_extinction/extinguish.rst>
   Fitting extinction curves <dust_extinction/fit_extinction.rst>
   References <dust_extinction/references.rst>

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
    wavelengths = np.logspace(np.log10(1000), np.log10(3e4), num=1000)*u.AA
    bb_lam = BlackBody(10000*u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
    spectrum = bb_lam(wavelengths)

Define a model, specifically the F99 model with an R(V) = 3.1.

.. code-block:: python

    from dust_extinction.parameter_averages import G23

    # define the model
    ext = G23(Rv=3.1)

Extinguish (redden) a spectrum with a screen of F99 dust with an E(B-V) of 0.5.
Can also specify the dust column with Av (this case equivalent to Av = 0.5*Rv =
1.55).

.. code-block:: python

    # extinguish (redden) the spectrum
    spectrum_ext = spectrum*ext.extinguish(wavelengths, Ebv=0.5)

Unextinguish (deredden) a spectrum with a screen of F99 dust with the
equivalent A(V) column.

.. code-block:: python

    # unextinguish (deredden) the spectrum
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
    # wavelengths between 1000 and 30000 A
    wavelengths = np.logspace(np.log10(1000), np.log10(3e4), num=1000)*u.AA
    bb_lam = BlackBody(10000*u.K, scale=1.0 * u.erg / (u.cm ** 2 * u.AA * u.s * u.sr))
    spectrum = bb_lam(wavelengths)

    # extinguish (redden) the spectrum
    spectrum_ext = spectrum*ext.extinguish(wavelengths, Ebv=0.5)

    # unextinguish (deredden) the spectrum
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
`Astropy`_.  The following pages will help you get started with contributing
fixes, code, or documentation (no git or GitHub experience necessary):

* `How to make a code contribution <https://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_

* `Coding Guidelines <https://docs.astropy.org/en/stable/development/codeguide.html>`_

* `Developer Documentation <https://docs.astropy.org/en/latest/index_dev.html>`_

For the complete list of contributors please see the `dust_extinction
contributors page on Github
<https://github.com/karllark/dust_extinction/graphs/contributors>`_.

Reference API
=============

.. toctree::
   :maxdepth: 2

   dust_extinction/reference_api.rst
