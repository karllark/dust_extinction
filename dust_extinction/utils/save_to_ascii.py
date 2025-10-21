import numpy as np
import astropy.units as u
from astropy.table import QTable
from dust_extinction.parameter_averages import G23
import matplotlib.pyplot as plt

if __name__ == '__main__':

    Rvs = np.array(range(34)) * 0.1 + 2.3
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
        cmod = G23(Rv=cRv)
        otab.meta["keywords"]["Rv"] = {"value": cRv}
        otab["AlambdaAv"] = cmod(waves)
        otab.write(f"dust_extinction_G23_Rv_{cRv:.1f}.ipac", format="ascii.ipac", overwrite=True)

        plt.plot(otab["wavelength"], otab["AlambdaAv"])

    plt.xscale("log")
    plt.yscale("log")
    plt.show()
