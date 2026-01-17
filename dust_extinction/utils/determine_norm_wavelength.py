import numpy as np
import astropy.units as u

from dust_extinction.tests.helpers import ave_models, param_ave_models, grain_models


print("V band normalization wavelength = A(l)/A(V) = 1")
print("B band normalization wavelength = A(l)/A(V) = 1 + 1/R(V)")
optwaves = np.arange(0.7, 0.4, -0.01) * u.micron
twave = 1 / 0.55
for cmodel in ave_models + param_ave_models + grain_models:
    cmod = cmodel()
    if (twave > cmod.x_range[0]) & (twave < cmod.x_range[1]):
        mvals = cmod(optwaves)
        nwave = np.interp([1.0], mvals, optwaves)
        nwave2 = np.interp([1.0 + 1 / cmod.Rv], mvals, optwaves)
        Rv = cmod.Rv
        if not isinstance(Rv, float):
            Rv = Rv.value
        print(
            f"| {cmod.__class__.__name__:12.12s} | {nwave.squeeze().value:18.4f} | {nwave2.squeeze().value:27.4f} | {Rv:.2f} |"
        )
        print(
            "+--------------+--------------------+-----------------------------+------+"
        )
    else:
        print(cmod.__class__.__name__, " wave coverage doesn't include V band")

print("")
print("R(V) relations where A(l)/A(V) is the same for Rv = 2.5 & 5.0")

for cmodel in param_ave_models[0:-1]:

    cmod1 = cmodel(Rv=2.5)
    cmod2 = cmodel(Rv=5.0)

    if (twave > cmod1.x_range[0]) & (twave < cmod1.x_range[1]):

        mvals1 = cmod1(optwaves)
        mvals2 = cmod2(optwaves)

        # find the wavelength where the difference in the curves is zero
        diff = mvals1 - mvals2
        nwave = np.interp([0.0], diff, optwaves)
        print(cmod1.__class__.__name__, nwave, cmod1(nwave))


# test the G23 original
cmod1 = cmodel(Rv=2.5, renorm=False)
cmod2 = cmodel(Rv=5.0, renorm=False)

mvals1 = cmod1(optwaves)
mvals2 = cmod2(optwaves)

# find the wavelength where the difference in the curves is zero
diff = mvals1 - mvals2
nwave = np.interp([0.0], diff, optwaves)
print(cmod1.__class__.__name__, "orig", nwave, cmod1(nwave))
