# routine to compute the CCM89 R(V) dependent extinction curve
# some code adapted from google code extinct.py by Erik Tollerud
#  adaption needed as we are directly interested in the extinction curve
#  not in dereddening a spectrum (forward instead of reverse modeling)
#  and some changes were needed to the limits

import numpy as np

from astropy.modeling import Model, Parameter

class CCM89(Model):
    Rv = Parameter(description="R(V) = A(V)/E(B-V) = " \
                   + "total-to-selective extinction",
                   default=3.1,
                   min=2.0, max=6.0)
    #Rv_range = [2.0,6.0]

    #@Rv.validator
    #def Rv(self, value):
    #    if not (Rv_range[0] <= value <= Rv_range[1]):
    #        raise InputParameterError("parameter Rv must be between "
    #                                  + string(Rv_range[0])
    #                                  + " and "
    #                                  + string(Rv_range[1])
    #                                  + '.')
    
    @staticmethod
    def evaluate(x, Rv):
        
        n_x = len(x)
        a = np.zeros(n_x)
        b = np.zeros(n_x)

        ir_indxs = np.where(np.logical_and(0.3 <= x,x < 1.1))
        opt_indxs = np.where(np.logical_and(1.1 <= x,x < 3.3))
        nuv_indxs = np.where(np.logical_and(3.3 <= x,x <= 8.0))
        fnuv_indxs = np.where(np.logical_and(5.9 <= x,x <= 8))
        fuv_indxs = np.where(np.logical_and(8 < x,x <= 10))
    
        # Infrared
        y = x[ir_indxs]**1.61
        a[ir_indxs] = .574*y
        b[ir_indxs] = -0.527*y
    
        # NIR/optical
        y = x[opt_indxs] - 1.82
        a[opt_indxs] = np.polyval((.32999, -.7753, .01979, .72085, -.02427,
                                   -.50447, .17699, 1), y)
        b[opt_indxs] = np.polyval((-2.09002, 5.3026, -.62251, -5.38434,
                                   1.07233, 2.28305, 1.41338, 0), y)
    
        # NUV
        a[nuv_indxs] = 1.752-.316*x[nuv_indxs] \
                       - 0.104/((x[nuv_indxs] - 4.67)**2 + .341)
        b[nuv_indxs] = -3.09 + \
                       1.825*x[nuv_indxs] \
                       + 1.206/((x[nuv_indxs] - 4.62)**2 + .263)
        
        # far-NUV
        y = x[fnuv_indxs] - 5.9
        a[fnuv_indxs] += -.04473*(y**2) - .009779*(y**3)
        # sign changed on both from extinct.py where it was incorrect
        b[fnuv_indxs] += .2130*(y**2) + .1207*(y**3)  
        
        # FUV
        y = x[fuv_indxs] - 8.0
        a[fuv_indxs] = np.polyval((-.070, .137, -.628, -1.073), y)
        b[fuv_indxs] = np.polyval((.374, -.42, 4.257, 13.67), y)

        print(a,b)
        
        # return Al/Av
        return a + b/Rv
        
