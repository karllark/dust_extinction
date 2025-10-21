import numpy as np
import astropy.units as u
from astropy.table import QTable
from dust_extinction.parameter_averages import G23
import matplotlib.pyplot as plt

if __name__ == '__main__':

    Rv_range = [2.0, 6.51]
    n_Rvs = int(np.round((Rv_range[1] - Rv_range[0]) / 0.1) + 1.0)
    print("# Rvs", n_Rvs)
    Rvs = np.array(range(n_Rvs)) * 0.1 + Rv_range[0]
    print(Rvs)
    # Rvs = [2.3, 2.7, 3.1, 3.5, 4.0, 4.5, 5.0, 5.6]
    waves = np.logspace(np.log10(0.0912), np.log10(32.), num=1000) * u.micron

    otab = QTable()
    otab.meta["keywords"] = {}
    otab.meta["keywords"]["origin"] = {"value": "dust_extinction python package"}
    otab.meta["keywords"]["model"] = {"value": "G23"}
    otab.meta["keywords"]["author"] = {"value": "Karl D. Gordon"}
    otab["wavelength"] = waves
    for cRv in Rvs:
        cmod = G23()
        # override Rv_range to allow extrapolation - here be dragons
        #  Above Rv=6.3, FUV extinction becomes lower at shorter wavelengths
        #  Below Rv=2.2, MIR before 10um silicates goes unrealistically small
        cmod.Rv_range = Rv_range
        cmod.Rv = cRv
        otab.meta["keywords"]["Rv"] = {"value": cRv}
        otab["AlambdaAv"] = cmod(waves)
        otab.write(f"dust_extinction_G23_Rv_{cRv:.1f}.ipac", format="ascii.ipac", overwrite=True)

        plt.plot(otab["wavelength"], otab["AlambdaAv"])

    plt.xscale("log")
    plt.yscale("log")
    plt.show()
