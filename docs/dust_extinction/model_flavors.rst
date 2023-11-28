#############
Model Flavors
#############

There are three different types of models: average, R(V)+ dependent prediction,
and shape fitting.

Average models
==============

   These models provide averages from the literature with the ability to
   interpolate between the observed data points.
   Models are provided for the Milky Way for the optical (Bastiaansen 1992),
   ultraviolet through near-infrared
   (Gordon, Cartlege, & Clayton 2009) and near- and mid-infrared
   (Rieke & Lebofsky 1985; Indebetouw et al. 2005; Chiar & Tielens 2006; Fritz et al. 2011)
   and the Magellanic Clouds (Gordon et al. 2003).

   For the Milky Way for the ultraviolet through near-infrared,
   one of the R(V) dependent models with R(V) = 3.1
   (see next section) is often used for the Milky Way 'average'.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import astropy.units as u

   from dust_extinction.averages import (GCC09_MWAvg,
                                         B92_MWAvg,
                                         G03_SMCBar,
                                         G03_LMCAvg,
					                               G03_LMC2)

   fig, ax = plt.subplots()

   # generate the curves and plot them
   x = np.arange(0.3,11.0,0.1)/u.micron

   models = [GCC09_MWAvg, B92_MWAvg, G03_SMCBar, G03_LMCAvg, G03_LMC2]

   for cmodel in models:
      ext_model = cmodel()
      indxs, = np.where(np.logical_and(
         x.value >= ext_model.x_range[0],
         x.value <= ext_model.x_range[1]))
      yvals = ext_model(x[indxs])
      ax.plot(1./x[indxs], yvals, label=ext_model.__class__.__name__)

   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())

   ax.set_xlabel(r'$\lambda$ [$\mu$m]')
   ax.set_ylabel(r'$A(\lambda)/A(V)$')
   ax.set_title('Ultraviolet to Near-Infrared Models')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()


.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib.ticker import ScalarFormatter
  import astropy.units as u

  from dust_extinction.averages import (RL85_MWGC,
                                        RRP89_MWGC,
                                        I05_MWAvg,
                                        CT06_MWLoc,
                                        CT06_MWGC,
                                        F11_MWGC,
                                        G21_MWAvg,
                                        D22_MWAvg)

  fig, ax = plt.subplots()

  # generate the curves and plot them
  x = 1.0 / (np.arange(1.0, 40.0 ,0.1) * u.micron)

  models = [RL85_MWGC, RRP89_MWGC, I05_MWAvg, CT06_MWLoc, CT06_MWGC,
            F11_MWGC, G21_MWAvg, D22_MWAvg]

  for cmodel in models:
    ext_model = cmodel()
    indxs, = np.where(np.logical_and(
       x.value >= ext_model.x_range[0],
       x.value <= ext_model.x_range[1]))
    yvals = ext_model(x[indxs])
    ax.plot(1.0 / x[indxs], yvals, label=ext_model.__class__.__name__)

  ax.set_xscale('log')
  ax.xaxis.set_major_formatter(ScalarFormatter())

  ax.set_xlabel(r'$\lambda$ [$\mu$m]')
  ax.set_ylabel(r'$A(\lambda)/A(V)$')
  ax.set_title('Near- to Mid-Infrared Models')

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
   (Maiz Apellaniz et al. 2014), was developed for the LMC but
   has been shown valid elsewhere in the Milky Way.

   In addition, the (R(V), f_A) two parameter relationship from
   Gordon et al. (2016) is included.  This model is based on the average
   behavior of extinction in the Milky Way, Large Magellanic Cloud, and
   Small Magellanic Cloud.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import astropy.units as u

   from dust_extinction.parameter_averages import (CCM89, O94, F99, F04,
                                                   VCG04, GCC09, M14, F19, D22,
                                                   G23)

   fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

   # generate the curves and plot them
   x = np.arange(1./30., 1./0.0912, 0.001)/u.micron

   Rv = 3.1

   models = [CCM89, O94, F99, F04, VCG04, GCC09, M14, F19, D22, G23]

   for cmodel in models:
      ext_model = cmodel(Rv=Rv)
      indxs, = np.where(np.logical_and(
         x.value >= ext_model.x_range[0],
         x.value <= ext_model.x_range[1]))
      yvals = ext_model(x[indxs])
      ax[0].plot(1./x[indxs], yvals, label=ext_model.__class__.__name__)
      ax[1].plot(1./x[indxs], yvals, label=ext_model.__class__.__name__)

   for iax in ax:
      iax.set_xscale('log')
      iax.xaxis.set_major_formatter(ScalarFormatter())

      iax.set_xlabel(r'$\lambda$ [$\mu$m]')
      iax.set_ylabel(r'$A(\lambda)/A(V)$')

   ax[0].set_title(f'UV-NIR R(V) = {Rv}')
   ax[0].set_xlim(0.08, 3.0)
   ax[1].set_title(f'NIR-MIR R(V) = {Rv}')
   ax[1].set_xlim(1.0, 32.0)
   ax[1].set_ylim(0.0, 0.50)

   ax[0].legend(loc='best')
   ax[1].legend(loc='best')
   plt.tight_layout()
   plt.show()


.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import astropy.units as u

   from dust_extinction.parameter_averages import (CCM89, O94, F99, F04,
                                                   VCG04, GCC09, M14, F19, D22,
                                                   G23)

   fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

   # generate the curves and plot them
   x = np.arange(1./32., 1./0.0912, 0.001)/u.micron

   Rv = 2.5

   models = [CCM89, O94, F99, F04, VCG04, GCC09, M14, F19, D22, G23]

   for cmodel in models:
      ext_model = cmodel(Rv=Rv)
      indxs, = np.where(np.logical_and(
         x.value >= ext_model.x_range[0],
         x.value <= ext_model.x_range[1]))
      yvals = ext_model(x[indxs])
      ax[0].plot(1./x[indxs], yvals, label=ext_model.__class__.__name__)
      ax[1].plot(1./x[indxs], yvals, label=ext_model.__class__.__name__)

   for iax in ax:
      iax.set_xscale('log')
      iax.xaxis.set_major_formatter(ScalarFormatter())

      iax.set_xlabel(r'$\lambda$ [$\mu$m]')
      iax.set_ylabel(r'$A(\lambda)/A(V)$')

   ax[0].set_title(f'UV-NIR R(V) = {Rv}')
   ax[0].set_xlim(0.08, 3.0)
   ax[1].set_title(f'NIR-MIR R(V) = {Rv}')
   ax[1].set_xlim(1.0, 32.0)
   ax[1].set_ylim(0.0, 0.50)

   ax[0].legend(loc='best')
   ax[1].legend(loc='best')
   plt.tight_layout()
   plt.show()


.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import astropy.units as u

   from dust_extinction.parameter_averages import (CCM89, O94, F99, F04,
                                                   VCG04, GCC09, M14, F19, D22,
                                                   G23)

   fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

   # generate the curves and plot them
   x = np.arange(1./32., 1./0.0912, 0.001)/u.micron

   Rv = 5.5

   models = [CCM89, O94, F99, F04, VCG04, GCC09, M14, F19, D22, G23]

   for cmodel in models:
      ext_model = cmodel(Rv=Rv)
      indxs, = np.where(np.logical_and(
         x.value >= ext_model.x_range[0],
         x.value <= ext_model.x_range[1]))
      yvals = ext_model(x[indxs])
      ax[0].plot(1./x[indxs], yvals, label=ext_model.__class__.__name__)
      ax[1].plot(1./x[indxs], yvals, label=ext_model.__class__.__name__)

   for iax in ax:
      iax.set_xscale('log')
      iax.xaxis.set_major_formatter(ScalarFormatter())

      iax.set_xlabel(r'$\lambda$ [$\mu$m]')
      iax.set_ylabel(r'$A(\lambda)/A(V)$')

   ax[0].set_title(f'UV-NIR R(V) = {Rv}')
   ax[0].set_xlim(0.08, 3.0)
   ax[1].set_title(f'NIR-MIR R(V) = {Rv}')
   ax[1].set_xlim(1.0, 32.0)
   ax[1].set_ylim(0.0, 0.50)

   ax[0].legend(loc='best')
   ax[1].legend(loc='best')
   plt.tight_layout()
   plt.show()

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import astropy.units as u

   from dust_extinction.parameter_averages import G16

   fig, ax = plt.subplots()

   # temp model to get the correct x range
   text_model = G16()

   # generate the curves and plot them
   x = np.arange(text_model.x_range[0], text_model.x_range[1],0.1)/u.micron

   Rvs = [2.0, 3.0, 4.0, 5.0, 6.0]
   for cur_Rv in Rvs:
      ext_model = G16(RvA=cur_Rv, fA=1.0)
      ax.plot(1./x,ext_model(x),label=r'$R_A(V) = ' + str(cur_Rv) + '$')

   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())

   ax.set_xlabel(r'$\lambda$ [$\mu$m]')
   ax.set_ylabel(r'$A(\lambda)/A(V)$')

   ax.set_title('G16; $f_A = 1.0$; $R(V)_A$ variable')

   ax.legend(loc='best', title=r'$f_A = 1.0$')
   plt.tight_layout()
   plt.show()

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
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
      ax.plot(1./x,ext_model(x),label=r'$f_A = ' + str(cur_fA) + '$')

   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())

   ax.set_xlabel(r'$\lambda$ [$\mu$m]')
   ax.set_ylabel(r'$A(\lambda)/A(V)$')

   ax.set_title('G16; $f_A$ variable; $R(V)_A = 3.1$')

   ax.legend(loc='best', title=r'$R_A(V) = 3.1$')
   plt.tight_layout()
   plt.show()


Grain models
============

   These models provide literature grain models
   interpolated between the computed data points.
   These dust grain models are based on fitting observed extinction curves and
   other observed properties of dust (e.g., abundances, IR emission).
   Models are provided for the Milky Way calculated for the X-ray to the submm.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dust_extinction.grain_models import DBP90, WD01, D03, ZDA04, C11, J13, HD23

   fig, ax = plt.subplots()

   # generate the curves and plot them
   lam = np.logspace(-4.0, 5.0, num=1000)
   x = (1.0 / lam) / u.micron

   models = [DBP90,
             WD01, WD01, WD01,
             D03, D03, D03,
             ZDA04,
             C11, J13,
             HD23]
   modelnames = ["MWRV31",
                 "MWRV31", "MWRV40", "MWRV55",
                 "MWRV31", "MWRV40", "MWRV55",
                 "BARE-GR-S",
                 "MWRV31", "MWRV31",
                 "MWRV31"]

   for cmodel, cname in zip(models, modelnames):
      ext_model = cmodel(cname)

      indxs, = np.where(np.logical_and(
         x.value >= ext_model.x_range[0],
         x.value <= ext_model.x_range[1]))
      yvals = ext_model(x[indxs])
      ax.plot(lam[indxs], yvals, label=f"{ext_model.__class__.__name__}  {cname}")

   ax.set_xlabel('$\lambda$ [$\mu m$]')
   ax.set_ylabel(r'$A(\lambda)/A(V)$')
   ax.set_title('Grain Models')

   ax.set_xscale('log')
   ax.set_yscale('log')

   ax.set_title('Milky Way')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()


.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import astropy.units as u

   from dust_extinction.grain_models import DBP90, WD01, D03, ZDA04, C11, J13, HD23

   fig, ax = plt.subplots()

   # generate the curves and plot them
   lam = np.logspace(np.log10(0.0912), np.log10(50.), num=1000)
   x = (1.0 / lam) / u.micron

   models = [DBP90,
             WD01, WD01, WD01,
             D03, D03, D03,
             ZDA04,
             C11, J13,
             HD23]
   modelnames = ["MWRV31",
                 "MWRV31", "MWRV40", "MWRV55",
                 "MWRV31", "MWRV40", "MWRV55",
                 "BARE-GR-S",
                 "MWRV31", "MWRV31",
                 "MWRV31"]

   for cmodel, cname in zip(models, modelnames):
      ext_model = cmodel(cname)

      indxs, = np.where(np.logical_and(
         x.value >= ext_model.x_range[0],
         x.value <= ext_model.x_range[1]))
      yvals = ext_model(x[indxs])
      ax.plot(lam[indxs], yvals, label=f"{ext_model.__class__.__name__}  {cname}")

   ax.set_xlabel('$\lambda$ [$\mu m$]')
   ax.set_ylabel(r'$A(\lambda)/A(V)$')
   ax.set_title('Grain Models')

   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())
   ax.set_yscale('log')

   ax.set_title('Milky Way - Ultraviolet to Mid-Infrared')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

.. plot::

  import numpy as np
  import matplotlib.pyplot as plt
  import astropy.units as u

  from dust_extinction.grain_models import WD01

  fig, ax = plt.subplots()

  # generate the curves and plot them
  lam = np.logspace(-4.0, 4.0, num=1000)
  x = (1.0 / lam) / u.micron

  models = [WD01, WD01, WD01]
  modelnames = ["LMCAvg", "LMC2", "SMCBar"]

  for cmodel, cname in zip(models, modelnames):
     ext_model = cmodel(cname)

     indxs, = np.where(np.logical_and(
        x.value >= ext_model.x_range[0],
        x.value <= ext_model.x_range[1]))
     yvals = ext_model(x[indxs])
     ax.plot(lam[indxs], yvals, label=f"{ext_model.__class__.__name__}  {cname}")

  ax.set_xlabel('$\lambda$ [$\mu m$]')
  ax.set_ylabel(r'$A(\lambda)/A(V)$')
  ax.set_title('Grain Models')

  ax.set_xscale('log')
  ax.set_yscale('log')

  ax.set_title('LMC & SMC')

  ax.legend(loc='best')
  plt.tight_layout()
  plt.show()


Shape fitting models
====================

   These models are used to fit the detailed shape of dust extinction curves.
   The FM90 (Fitzpatrick & Mass 1990) model uses 6 parameters to fit the
   shape of the ultraviolet extinction.
   Note there are two forms of the FM90 model, FM90 that implements the model
   as published and FM90_B3 that B3 = C3/gamma^2 as the explicit amplitude of
   the 2175 A bump (easier to interpret).
   The P92 (Pei 1992) uses 19 parameters to fit the shape of the X-ray to
   far-infrared extinction.
   The G21 (Gordon et al. 2021) models uses 10 parameters to fit the shape
   of the NIR/MIR 1-40 micron extinction.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import astropy.units as u

   from dust_extinction.shapes import FM90

   fig, ax = plt.subplots()

   # generate the curves and plot them
   x = np.arange(3.8,11.0,0.1)/u.micron

   ext_model = FM90()
   ax.plot(1./x,ext_model(x),label='total')

   ext_model = FM90(C3=0.0, C4=0.0)
   ax.plot(1./x,ext_model(x),label='linear term')

   ext_model = FM90(C1=0.0, C2=0.0, C4=0.0)
   ax.plot(1./x,ext_model(x),label='bump term')

   ext_model = FM90(C1=0.0, C2=0.0, C3=0.0)
   ax.plot(1./x,ext_model(x),label='FUV rise term')

   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())
   ax.xaxis.set_minor_formatter(ScalarFormatter())
   ax.set_xlabel(r'$\lambda$ [$\mu$m]')
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
   ax.set_ylabel(r'$A(\lambda)/A(V)$')

   ax.set_title('P92')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import ScalarFormatter
   import astropy.units as u

   from dust_extinction.shapes import G21

   fig, ax = plt.subplots()

   # generate the curves and plot them
   lam = np.logspace(np.log10(1.01), np.log10(39.9), num=1000)
   x = (1.0/lam)/u.micron

   ext_model = G21()
   ax.plot(1/x,ext_model(x),label='total')

   ext_model = G21(sil1_amp=0.0, sil2_amp=0.0)
   ax.plot(1./x,ext_model(x),label='power-law only')

   ext_model = G21(sil2_amp=0.0)
   ax.plot(1./x,ext_model(x),label='power-law+sil1 only')

   ext_model = G21(sil1_amp=0.0)
   ax.plot(1./x,ext_model(x),label='power-law+sil2 only')

   ax.set_xscale('log')
   ax.xaxis.set_major_formatter(ScalarFormatter())
   ax.set_yscale('log')

   ax.set_xlabel('$\lambda$ [$\mu$m]')
   ax.set_ylabel(r'$A(\lambda)/A(V)$')

   ax.set_title('G21')

   ax.legend(loc='best')
   plt.tight_layout()
   plt.show()
