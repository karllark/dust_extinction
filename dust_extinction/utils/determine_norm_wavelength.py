import numpy as np
import astropy.units as u

from dust_extinction.tests.helpers import ave_models, param_ave_models, grain_models


optwaves = np.arange(0.7, 0.4, -0.01) * u.micron
twave = 1 / 0.55
for cmodel in ave_models + param_ave_models + grain_models:
    cmod = cmodel()
    if (twave > cmod.x_range[0]) & (twave < cmod.x_range[1]):
        mvals = cmod(optwaves)
        nwave = np.interp([1.0], mvals, optwaves)
        print(cmod.__class__.__name__, nwave)
    else:
        print(cmod.__class__.__name__, " wave coverage doesn't include V band")
