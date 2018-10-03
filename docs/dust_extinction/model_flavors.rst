#############
Model Flavors
#############

There are three different types of models: average, R(V)+ dependent prediction,
and shape fitting.

Average models
==============

   These models provide averages from the literature with the ability to
   interpolate between the observed data points.
   Models are provided for the Milky Way (Gordon, Cartlege, & Clayton 2009)
   and the Magellanic Clouds (Gordon et al. 2003).

   For the Milky Way, one of the R(V) dependent models with R(V) = 3.1
   (see next section) can also be used for the Milky Way 'average'.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.averages import (GCC09_MWAvg,
                                         G03_SMCBar,
                                         G03_LMCAvg,
					                               G03_LMC2)

   fig, ax = plt.subplots()

   # generate the curves and plot them
   x = np.arange(0.3,10.0,0.1)/u.micron
   ext_model = GCC09_MWAvg()
   ax.plot(x,ext_model(x),label='GCC09 MWAvg')

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

R(V) (+ other variables) dependent prediction models
====================================================

   These models provide predictions of the shape of the dust extinction
   given input variable(s).

   The R(V) dependent models include CCM89 the original such model
   (Cardelli, Clayton, and Mathis 1989), the O94 model that updates the
   optical portion of the CCM89 model (O'Donnell 1994), and the F99 model
   (Fitzpatrick 1999) updated as F04 (Fitzpatrick 2004),
   These models are based on the average
   behavior of extinction in the Milky Way as a function of R(V).
   The M14 model refines the optical portion of the CCM89 model
   (Ma\’{\i}z Apell\’aniz et al. 2014), was developed for the LMC but
   has been shown valid elsewhere in the Milky Way.

   In addition, the (R(V), f_A) two parameter relationship from
   Gordon et al. (2016) is included.  This model is based on the average
   behavior of extinction in the Milky Way, Large Magellanic Cloud, and
   Small Magellanic Cloud.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.parameter_averages import (CCM89, O94, F99, F04, M14)

   fig, ax = plt.subplots()

   # generate the curves and plot them
   x = np.arange(0.5,10.0,0.1)/u.micron

   Rv = 3.1

   ext_model = CCM89(Rv=Rv)
   ax.plot(x,ext_model(x),label='CCM89')

   ext_model = O94(Rv=Rv)
   ax.plot(x,ext_model(x),label='O94')

   ext_model = F99(Rv=Rv)
   ax.plot(x,ext_model(x),label='F99')

   ext_model = F04(Rv=Rv)
   ax.plot(x,ext_model(x),label='F04')

   ext_model = M14(Rv=Rv)
   ax.plot(x,ext_model(x),label='M14')

   ax.set_xlabel('$x$ [$\mu m^{-1}$]')
   ax.set_ylabel('$A(x)/A(V)$')

   ax.set_title('R(V) = 3.1')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.parameter_averages import (CCM89, O94, F99, F04, M14)

   fig, ax = plt.subplots()

   # generate the curves and plot them
   x = np.arange(0.5,10.0,0.1)/u.micron

   Rv = 2.0

   ext_model = CCM89(Rv=Rv)
   ax.plot(x,ext_model(x),label='CCM89')

   ext_model = O94(Rv=Rv)
   ax.plot(x,ext_model(x),label='O94')

   ext_model = F99(Rv=Rv)
   ax.plot(x,ext_model(x),label='F99')

   ext_model = F04(Rv=Rv)
   ax.plot(x,ext_model(x),label='F04')

   ext_model = M14(Rv=Rv)
   ax.plot(x,ext_model(x),label='M14')

   ax.set_xlabel('$x$ [$\mu m^{-1}$]')
   ax.set_ylabel('$A(x)/A(V)$')

   ax.set_title('R(V) = 2.0')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()


.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.parameter_averages import (CCM89, O94, F99, F04, M14)

   fig, ax = plt.subplots()

   # generate the curves and plot them
   x = np.arange(0.5,10.0,0.1)/u.micron

   Rv = 5.5

   ext_model = CCM89(Rv=Rv)
   ax.plot(x,ext_model(x),label='CCM89')

   ext_model = O94(Rv=Rv)
   ax.plot(x,ext_model(x),label='O94')

   ext_model = F99(Rv=Rv)
   ax.plot(x,ext_model(x),label='F99')

   ext_model = F04(Rv=Rv)
   ax.plot(x,ext_model(x),label='F04')

   ext_model = M14(Rv=Rv)
   ax.plot(x,ext_model(x),label='M14')

   ax.set_xlabel('$x$ [$\mu m^{-1}$]')
   ax.set_ylabel('$A(x)/A(V)$')

   ax.set_title('R(V) = 5.5')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.parameter_averages import G16

   fig, ax = plt.subplots()

   # temp model to get the correct x range
   text_model = G16()

   # generate the curves and plot them
   x = np.arange(text_model.x_range[0], text_model.x_range[1],0.1)/u.micron

   Rvs = ['2.0','3.0','4.0','5.0','6.0']
   for cur_Rv in Rvs:
      ext_model = G16(RvA=cur_Rv, fA=1.0)
      ax.plot(x,ext_model(x),label=r'$R_A(V) = ' + str(cur_Rv) + '$')

   ax.set_xlabel('$x$ [$\mu m^{-1}$]')
   ax.set_ylabel('$A(x)/A(V)$')

   ax.set_title('G16; $f_A = 1.0$; $R(V)_A$ variable')

   ax.legend(loc='best', title=r'$f_A = 1.0$')
   plt.tight_layout()
   plt.show()

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.parameter_averages import G16

   fig, ax = plt.subplots()

   # temp model to get the correct x range
   text_model = G16()

   # generate the curves and plot them
   x = np.arange(text_model.x_range[0], text_model.x_range[1],0.1)/u.micron

   fAs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
   for cur_fA in fAs:
      ext_model = G16(RvA=3.1, fA=cur_fA)
      ax.plot(x,ext_model(x),label=r'$f_A = ' + str(cur_fA) + '$')

   ax.set_xlabel('$x$ [$\mu m^{-1}$]')
   ax.set_ylabel('$A(x)/A(V)$')

   ax.set_title('G16; $f_A$ variable; $R(V)_A = 3.1$')

   ax.legend(loc='best', title=r'$R_A(V) = 3.1$')
   plt.tight_layout()
   plt.show()


Shape fitting models
====================

   These models are used to fit the detailed shape of dust extinction curves.
   The FM90 (Fitzpatrick & Mass 1990) model uses 6 parameters to fit the
   shape of the ultraviolet extinction.
   The P92 (Pei 1992) uses 19 parameters to fit the shape of the X-ray to
   far-infrared extinction.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.shapes import FM90

   fig, ax = plt.subplots()

   # generate the curves and plot them
   x = np.arange(3.8,8.6,0.1)/u.micron

   ext_model = FM90()
   ax.plot(x,ext_model(x),label='total')

   ext_model = FM90(C3=0.0, C4=0.0)
   ax.plot(x,ext_model(x),label='linear term')

   ext_model = FM90(C1=0.0, C2=0.0, C4=0.0)
   ax.plot(x,ext_model(x),label='bump term')

   ext_model = FM90(C1=0.0, C2=0.0, C3=0.0)
   ax.plot(x,ext_model(x),label='FUV rise term')

   ax.set_xlabel('$x$ [$\mu m^{-1}$]')
   ax.set_ylabel('$E(\lambda - V)/E(B - V)$')

   ax.set_title('FM90')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.shapes import P92

   fig, ax = plt.subplots()

   # generate the curves and plot them
   lam = np.logspace(-3.0, 3.0, num=1000)
   x = (1.0/lam)/u.micron

   ext_model = P92()
   ax.plot(1/x,ext_model(x),label='total')

   ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                   SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
   ax.plot(1./x,ext_model(x),label='BKG only')

   ext_model = P92(NUV_amp=0.0,
                   SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
   ax.plot(1./x,ext_model(x),label='BKG+FUV only')

   ext_model = P92(FUV_amp=0.,
                   SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
   ax.plot(1./x,ext_model(x),label='BKG+NUV only')

   ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                   SIL2_amp=0.0)
   ax.plot(1./x,ext_model(x),label='BKG+FIR+SIL1 only')

   ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                   SIL1_amp=0.0)
   ax.plot(1./x,ext_model(x),label='BKG+FIR+SIL2 only')

   ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                   SIL1_amp=0.0, SIL2_amp=0.0)
   ax.plot(1./x,ext_model(x),label='BKG+FIR only')

   ax.set_xscale('log')
   ax.set_yscale('log')

   ax.set_ylim(1e-3,10.)

   ax.set_xlabel('$\lambda$ [$\mu$m]')
   ax.set_ylabel('$A(x)/A(V)$')

   ax.set_title('P92')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()
