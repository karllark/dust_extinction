from astropy.modeling import (Fittable1DModel, Parameter)

__all__ = ['AxAvToExv']


class AxAvToExv(Fittable1DModel):
    """
    Model to convert from A(x)/A(V) to E(x-V)

    Paramters
    ---------
    Av : float
      dust column in A(V) [mag]
    """
    inputs = ('axav',)
    outputs = ('exv',)

    Av = Parameter(description="A(V)",
                   default=1.0, min=0.0)

    @staticmethod
    def evaluate(axav, Av):
        """
        AlAvToElv function

        Paramters
        ---------
        axav : np array (float)
           E(x-V)/E(B-V) values

        Returns
        -------
        exv : np array (float)
           E(x - V)
        """
        return (axav - 1.0)*Av

    @staticmethod
    def fit_deriv(axav, Av):
        """
        Derivatives of the AxAvtoElv function with respect to the parameters
        """
        # derivatives
        d_Av = (axav - 1.0)

        return [d_Av]
