.. _ExtvsAtt:

#############################
Extinction versus Attenuation
#############################

.. note:: All extinction curves are attenuation curves, but not all attenuation
          curves are extinction curves.

Extinction
==========

Interstellar dust extinction is the result of photons being absorbed or
scattered *out* of the line-of-sight by dust grains.

Extinction can be directly measured by observing a star with dust along the
line-of-sight. Knowledge of the intrinsic spectrum of the star from either an
observation of a similar star without foreground dust along the line-of-sight
or a stellar atmosphere model is used.

Extinction can be directly measured towards individual stars.  The specific
geometry is a star with a column of dust between the observer and the star. The
two effects present are dust grains absorbing photons or scattering photons out
of the line-of-sight.  Since the dust grains are not near the star, scattering
of photons from the star into the line-of-sight is very small  and can be safely
ignored.

Both dust absorption and scattering out of the line-of-sight are processes
that are directly proportional to the amount of dust along the line-of-sight.
As a result, the ratio of dust extinctions at two different wavelengths
does not vary with different amounts of otherwise identical dust.  In other words, extinction
curves normalized by dust column measured towards different different stars
with different amounts of identical dust grains are equivalent.  This is
illustrated below where the left plot shows the total extinction as a function
of wavelength and the right plot shows the same curves normalized by A(V).

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.averages import GCC09_MWAvg

   fig, ax = plt.subplots(ncols=2)

   # generate the curves and plot them
   x = np.arange(0.3,10.0,0.1)/u.micron
   ext_model = GCC09_MWAvg()

   ax[0].plot(1./x,ext_model(x)*0.5,label='A(V) = 0.5 mag')
   ax[0].plot(1./x,ext_model(x)*1.5,label='A(V) = 1.5 mag')
   ax[0].plot(1./x,ext_model(x)*2.5,label='A(V) = 2.5 mag')

   ax[1].plot(1./x,ext_model(x),label='A(V) = 0.5 mag')
   ax[1].plot(1./x,ext_model(x),label='A(V) = 1.5 mag')
   ax[1].plot(1./x,ext_model(x),label='A(V) = 2.5 mag')

   ax[0].set_title('Total Extinction')
   ax[1].set_title('Normalized Extinction')
   ax[0].set_xlabel('$\lambda$ [$\mu m$]')
   ax[1].set_xlabel('$\lambda$ [$\mu m$]')
   ax[0].set_ylabel('$A(\lambda)$')
   ax[1].set_ylabel('$A(\lambda)/A(V)$')
   ax[0].set_xscale('log')
   ax[1].set_xscale('log')
   ax[0].set_xlim(0.09,4.0)
   ax[1].set_xlim(0.09,4.0)
  
   ax[0].legend(loc='best')
   ax[1].legend(loc='best')
   plt.tight_layout()
   plt.show()

Note that there is dust scattered light throughout a galaxy due to the all the
stars and dust in that galaxy.  For our Galaxy, this is called the  Diffuse
Galactic Light (DGL). This overall scattered light is removed from observations
of a single star by the standard practice of subtracting a local background.
The overall scattered light is smooth on the usual spatial scales involved in
measuring a single star.

Attenuation
===========

Dust attenuation refers to the general impact on the spectrum of an object due
to the presence of dust.  In general, attenuation is used to indicate that the
geometry of the sources and dust in a system is more complex than a single star
with a foreground screen of dust.  Examples of such systems include dusty
galaxies (composed of many stars) and  stars with circumstellar dust.

Attenuation includes two additional effects not included in extinction. These are
scattering of photons into the observation beam and sources extinguished by
different columns of dust.  These two additional sources results in the ratio of
dust attenuations at two different wavelengths *varying* with different
amounts total system dust.  These two additional effects mean that the
measurement and/or theoretical calculation of attenuation is significantly more
complex than for extinction.

The separate package `dust_attenuation package
<http://dust-attenuation.readthedocs.io/>`_ exists to provide attenuation
models.
