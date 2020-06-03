"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"
Institute of Molecular Biology and Biotechnology (IMBB)
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
import numpy as np
from scipy.special import erf

def peak_height(B, s):
    smax = 1/float(s)
    B = (B-np.amin(B)) + 0.01
    rho_zero = (8*np.pi/B)*(-smax*np.exp(-B*smax**2/4)+np.sqrt(np.pi/B)*erf(np.sqrt(B)*smax/2))
    rho_zero = rho_zero /max(rho_zero)
    return(rho_zero)

