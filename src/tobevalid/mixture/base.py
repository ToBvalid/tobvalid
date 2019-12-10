# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:14:24 2019

@author: KavehB
"""

import numpy as np

class BaseMixture:
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """
    
    def __init__(self, n_modes, tol, max_iter):
        self.n_modes = n_modes
        self.tol = tol
        self.max_iter = max_iter
        self._loglike = 0
        self.mix = np.ones(self.n_modes)/self.n_modes
        self.nit = 0
 
    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        """
        if self.n_modes < 1:
            raise ValueError("Invalid value for 'n_modes': %d "
                             "EM requires at least one mode"
                             % self.n_modes)

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        # Check all the parameters values of the derived class
        self._check_parameters(X)

    
    def _check_parameters(self, X, **kwargs):
        pass
     
    def pdf(self, X):
        if isinstance(X, (np.ndarray)):
            return np.matmul(self._pdf(np.repeat(X[np.newaxis,...], self.n_modes, axis=0).T ), self.mix.T) 
         
        if isinstance(X, (int, float)):
            return np.matmul(self._pdf(X), self.mix.T)
        
        if isinstance(X, (list, tuple)) and all(isinstance(x, (int, float)) for x in X):
            return self.pdf(np.array(X))
        

        raise ValueError( "Expected numerical array or number, got {} instead.".format(X)) 
         
    
    def _pdf(self, X):
        pass

    
    def fit(self, X, **kwargs):
        self._check_X(X)
        self._check_parameters(X, **kwargs)
        
        self.data = X
        self.data_n = np.repeat(self.data[np.newaxis,...], self.n_modes, axis=0).T 
        
        self._init_parameters(**kwargs)
        
        self._converged_ = False
        
        lower_bound = -np.infty
        self.nit = 0
        for n_iter in np.arange(1, self.max_iter + 1):
           
            prev_lower_bound = lower_bound
            self._e_step()
            if not self._m_step():
                break
            
            lower_bound = self._loglike
    
            change = lower_bound - prev_lower_bound

            if n_iter > 1 and np.abs(change/prev_lower_bound) < self.tol:
                self._converged_ = True
                break
        self.nit = n_iter    
    def loglike(self):
        return self._loglike
    
        
    def _check_X(self, X):
        
        if not isinstance(X, (np.ndarray)):
            raise ValueError( "Expected ndarray, got {} instead.".format(X))
         
        if len(X.shape) != 1:
            raise ValueError( "Expected one-dimentional ndarray, got {} instead.".format(X))
        
        if not X.dtype in [np.float32, np.float64]:
            raise ValueError( "Expected numerical ndarray, got {} instead.".format(X))
    
    def _init_parameters(self, **kwargs):
        pass
    
        
    
    def _mix_values(self):   
        values = self._pdf(self.data_n)
        return values*self.mix
    
    def _calc_posterior_z(self):
        wp = self._mix_values()
        
        den = np.sum(wp, axis = 1)

        for i in np.arange(len(wp)):
            for j in np.arange(len(wp[i])):
                if wp[i][j] != 0:
                    wp[i][j] = wp[i][j]/den[i]            
        self.Z = wp
        
    def _e_step(self):
        self._calc_posterior_z()
        self.N = np.sum(self.Z, axis = 1)
        self.mix = self.N*(1/np.sum(self.N))
    
    def _m_step(self):
        pass
    
    