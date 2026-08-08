import numpy as np
import os

from dust_extinction.baseclasses import BaseExtModel
from astropy.modeling import Parameter
from scipy.interpolate import interp1d


class BF25(BaseExtModel):
    """
    JWST-based extinction curve expressed as A(lambda)/A(V).

    This curve is based on measured extinction ratios A(lambda)/A(F162M)
    for JWST/NIRCam filters, normalized to A(V) using
    the Draine (2003) dust model with R_V = 5.5 evaluated at the
    effective wavelength of the F162M filter.

    References
    ----------
    Bravo Ferres et al. (2025, A&A 704, A130)
    """
    
    x_range = [1 / 5.0, 1 / 1.0]


    obsdata_x = 1.0 / np.array(
        [1.20300720, 1.63424360, 1.85088840, 2.12136100, 3.61966750, 4.05162100, 4.70773970, 4.80950540])
    
    obsdata_axav = np.array(
        [0.32966511, 0.17916582, 0.14136183, 0.10875365, 0.05482474, 0.04443312, 0.04299980, 0.03762482])

    @classmethod
    def evaluate(cls, x):
        """
        Evaluate A(lambda)/A(V).

        Parameters
        ----------
        x : float or ndarray
            Inverse wavelength in units of micron^-1.

        Returns
        -------
        ndarray
            A(lambda)/A(V)
        """

        f = interp1d(cls.obsdata_x, cls.obsdata_axav)
        return f(x)
