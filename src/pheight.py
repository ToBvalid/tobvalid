# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:39:24 2019
"""
import numpy as np
import pandas as pd
from scipy.special import erf

def peak_height(B, s):
    smax = 1/float(s)
    B = (B-np.amin(B)) + 0.01
    rho_zero = (8*np.pi/B)*(-smax*np.exp(-B*smax**2/4)+np.sqrt(np.pi/B)*erf(np.sqrt(B)*smax/2))
    rho_zero = rho_zero /max(rho_zero)
    return(rho_zero)

def peak_height_f(s, in_file, out_file):
    df = pd.read_csv(in_file)
    in_data = df.x
    out_data = peak_height(in_data, s)
    out_df = pd.DataFrame({'x':out_data})
    out_df.to_csv(out_file)
	
