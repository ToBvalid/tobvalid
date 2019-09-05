# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:04:37 2019

@author: KavehB
"""
import numpy as np


class GMMResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    mixture : GaussianMixture
        The Gaussian Mixture.  
    success : bool
        Whether or not the optimizer exited successfully.
    loglike: float
        Value of loglike function.
    z: ndarray
        z values    
    nit : int
        Number of iterations performed by the optimizer.

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
    
    
class Gaussian:
    "Model univariate Gaussian"
    def __init__(self, mu, sigma):
        #mean and standard deviation
        self.mu = mu
        self.sigma = sigma
    #probability density function
    def pdf(self, datum):
        return pdf(datum, self.mu, self.sigma)
    #printing model values
    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)


class GaussianMixture():
    "Model mixture of N univariate Gaussians and their EM estimation"

    def __init__(self, mu, sigma, mix):
        self.mix = mix
        self.mu = mu
        self.sigma = sigma
        self.dist = [Gaussian(m, s) for (m, s) in zip(mu, sigma)]
        self.mode = len(mix)
    
    def pdf(self, datum):
        return np.sum([d.pdf(datum)*m for (d, m) in zip(self.dist, self.mix)])
    
    
    def __repr__(self):
        return ''.join(['GaussianMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['Mixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])

def pdf(datum, mu, sigma):
    "Probability of a data point given the current parameters"
    u = (datum - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma))) * np.exp(-u * u / 2)
    return y
    
def emgmm(d, mode, max_iter = 100, x_tol = 0.01):
    # Initialization
    data = np.array(d)
    loglike = 0.
    mix = np.array([1./mode]*mode)
    dist = None
    mu_min = np.min(data)
    mu_max = np.max(data)
    mu = np.array([mu_min +  (mu_max - mu_min)*(2.*i + 1)/(2.*mode) for i in range(mode)])
    sigma =   np.array([(mu_max-mu_min)/(mode*np.sqrt(12.)) for i in range(mode)])
    
    
    last_loglike = float('-inf')
  
    best_mix =   GMMResult({'loglike':loglike, "mixture":GaussianMixture(mu, sigma, mix), "nit":1, "success":False, "z":None})
    for i in np.arange(max_iter):
#        try:
        
        #E Step
        
                
        dist = np.vectorize(pdf)
        values = dist(np.repeat(data[np.newaxis,...], mode, axis=0).T, mu, sigma).T
        wp = np.multiply(values.T, mix)
        den = np.sum(wp, axis = 1)
        z = np.multiply(wp.T, np.reciprocal(den, dtype=float))
       
        #M Step
        N = np.sum(z, axis = 1)
        
        mu = np.sum(np.multiply(np.multiply(z, d).T ,  np.reciprocal(N, dtype=float)), axis = 0 )
       
        diff = np.repeat(data[np.newaxis,...], mode, axis=0).T  - mu  
        diffSquare = np.multiply(diff, diff)
        wdiffSquare = np.multiply(z.T, diffSquare)
        sigma = np.sqrt(np.sum(np.multiply(wdiffSquare, np.reciprocal(N, dtype=float)), axis = 0 ))
        mix = N*(1./np.sum(N))
        
#        loglike = (-0.5)*np.sum(np.sum(np.multiply(wdiffSquare, np.reciprocal(sigma**2, dtype = float)) + np.multiply(z.T, np.log(sigma**2)), axis = 0 )) 
        loglike = -np.sum(np.log(wp))
        cur_mix =   GMMResult({'loglike':loglike, "mixture":GaussianMixture(mu, sigma, mix), "nit":i+1, "success":True, "z":z})
        if(last_loglike > loglike):
            best_mix
        if abs((last_loglike - loglike)/loglike) < x_tol:
            return cur_mix
        best_mix = cur_mix
        last_loglike = loglike
#        except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
#            pass
   
    return GMMResult({'loglike':loglike, "mixture":GaussianMixture(mu, sigma, mix), "nit":max_iter, "success":True, "z":z})

#dist = np.vectorize(pdf)
#x = np.arange(10)
#values = dist(np.repeat(x[np.newaxis,...], 2, axis=0).T, [1, 1], [1, 2]).T
