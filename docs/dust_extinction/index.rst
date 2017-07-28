Interstellar Dust Extinction
============================

Interstellar dust extinction curves implemnted as astropy models

Uses the astropy affiliated package template

Flavors of Models
=================

There are three differnet types of models (to be completed).

1. Average models

   These models provide averages from the literature with the ability to
   interpolate between the observed data points.
   Models are provided for the Magellanic Clouds from Gordon et al. (2003).
   Models for the Milky Way still to be added (both UV/optical/NIR and IR).

.. plot::
      
   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u
   
   from dust_extinction.dust_extinction import (G03_SMCBar,
                                                G03_LMCAvg,
					        G03_LMC2)
      
   fig, ax = plt.subplots()
      
   # generate the curves and plot them
   x = np.arange(0.3,10.0,0.1)/u.micron
      
   ext_model = G03_SMCBar()
   ax.plot(x,ext_model(x),label='G03 SMCBar')

   ext_model = G03_LMCAvg()
   ax.plot(x,ext_model(x),label='G03 LMCAvg')
   
   ext_model = G03_LMC2()
   ax.plot(x,ext_model(x),label='G03 LMC2')
   
   ax.set_xlabel('$x$ [$\mu m^{-1}$]')
   ax.set_ylabel('$A(x)/A(V)$')
      
   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()
     
2. Shape fitting models

   These models are used to fit the detailed shape of dust extinction curves.
   
   - FM90
   - others needed (P92)
     
3. R(V) (+ other variables) dependent prediction models

   These models provide predictions of the shape of the dust extinction
   given input variable(s).

   - CCM89 [function of R(V)]
   - F99 [function of R(V)]
   - others needed (GCC09, G16, etc)

Repository
==========

Github: <https://github.com/karllark/dust_extinction>

Reference API
=============
.. toctree::
   :maxdepth: 1

.. automodapi:: dust_extinction.dust_extinction
		
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
