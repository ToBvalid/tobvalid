# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:26:01 2019

@author: KavehB
"""

import numpy as np

from .base import BaseMixture


class InverseGammaMixture(BaseMixture):
     def __init__(self, n_modes = 1, tol = 1e-3, max_iter = 100):
        BaseMixture.__init__(self, n_modes, tol, max_iter) 
     
     def _check_parameters(self, X, **kwargs):
        return
     
     def _init_parameters(self, **kwargs):
        mu_min = np.min(self.data)
        mu_max = np.max(self.data)
        self.mu = np.array([mu_min +  (mu_max - mu_min)*(2.*i + 1)/(2.*self.n_modes) for i in range(self.n_modes)])
        self.sigma =   np.ones(self.n_modes)*(mu_max-mu_min)/(self.n_modes*np.sqrt(12.))

        
     def _m_step(self):
        N = np.sum(self.Z, axis = 0)

        self.mu = np.sum((self.Z* self.data_n) *np.reciprocal(N, dtype=float), axis = 0 )

        diff = self.data_n - self.mu  
        diffSquare = diff*diff
        wdiffSquare = diffSquare*self.Z
        self.sigma = np.sqrt(np.sum(wdiffSquare*np.reciprocal(N, dtype=float), axis = 0 ))

        self.mix = N*(1./np.sum(N))
        
        wp = self._mix_values()
        self._loglike = -np.sum(np.log(wp[wp > 1e-07]))
        return True
        
     def _pdf(self, X):
         u = (X - self.mu) / np.abs(self.sigma)
         y = (1 / (np.sqrt(2 * np.pi) * np.abs(self.sigma))) * np.exp(-u * u / 2)
         return y