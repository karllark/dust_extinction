from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["AxAvToExv"]


class AxAvToExv(Fittable1DModel):
    """
    Model to convert from A(x)/A(V) to E(x-V)

    Parameters
    ----------
    Av : float
      dust column in A(V) [mag]
    """

    Av = Parameter(description="A(V)", default=1.0, min=0.0)

    @staticmethod
    def evaluate(axav, Av):
        """
        AlAvToElv function

        Parameters
        ----------
        axav : np array (float)
           E(x-V)/E(B-V) values

        Returns
        -------
        exv : np array (float)
           E(x - V)
        """
        return (axav - 1.0) * Av


# Removed 4 Sep 2019 as this code is never used due to this model
# only is ever used in a compound model, hence the analytic derivatives
# are never used.  Code never used or tested, better to not have it.
# The code is kept as a comment if in the future there is a need for it.
#
#    @staticmethod
#    def fit_deriv(axav, Av):
#        """
#        Derivatives of the AxAvtoElv function with respect to the parameters
#        """
#        # derivatives
#        d_Av = axav - 1.0
#
#        return [d_Av]
