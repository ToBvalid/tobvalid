# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:04:37 2019

@author: KavehB
"""

import numpy as np
from scipy.special import erf
from math import sqrt, log, exp, pi
import embase as embase
import scipy.optimize as sco

class Gaussian(embase.Distribution):
    "Model univariate Gaussian"
    def __init__(self, mu, sigma):
        #mean and standard deviation
        self.mu = mu
        self.sigma = sigma

    #probability density function
    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y
    
    def dpdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y*(self.mu - datum)/(self.sigma**2)
    

    def cdf(self, datum):
        return 0.5*(1 + erf((datum - self.mu)/(sqrt(2)*self.sigma)))
    #printing model values
    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)
    
    def size(self):
        return 2
    def setParams(self, params):
        self.mu = params[0]
        self.sigma = params[1]
    def getParams(self):
        return [self.mu, self.sigma]


class GaussianMixture(embase.Mixture):
    "Model mixture of N univariate Gaussians and their EM estimation"

    def __init__(self, data, n):
        super().__init__(data, n)
        mu_min=min(data)
        mu_max=max(data)
        self.dist = [Gaussian(0, 1) for i in range(self.mode)]
        self.log = [0]*self.mode
        for i in range(self.mode):
            self.dist[i].mu =  mu_min +  (mu_max - mu_min)*(2*i + 1)/(2*self.mode)
            self.dist[i].sigma = (mu_max-mu_min)/(self.mode*sqrt(12))
    
    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        # compute weights
        self.loglike = 0. # = log(p = 1)
        for datum in self.data:
            # unnormalized weights
            wp = [m*d.pdf(datum) for (m, d) in zip(self.mix, self.dist)]
           
            # compute denominator
            den = sum(wp)
            # normalize
            z = [w/den for w in wp]; 
            
            # yield weight tuple
            yield z

    def Mstep(self, weights):
        "Perform an M(aximization)-step"
        # compute denominators

        z = list(zip(* weights))
        N = [sum(z[i]) for i in range(self.mode)]
        for i in range(self.mode):
            self.dist[i].mu =  sum(w * d / N[i] for (w, d) in zip(z[i], self.data))
            self.dist[i].sigma = sqrt(sum(w * ((d - self.dist[i].mu) ** 2) for (w, d) in zip(z[i], self.data)) / N[i])
            self.mix[i] = N[i] / sum(N)
            self.log[i] += -1*sum((w * ((d - self.dist[i].mu) ** 2))/(self.dist[i].sigma**2) 
                + w*log(self.dist[i].sigma**2) for (w, d) in zip(z[i], self.data))/2
        self.loglike = sum(self.log)
        self.z = z

    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"
        self.Mstep(self.Estep())  
          
    def __repr__(self):
        return ''.join(['GaussianMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['Mixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])



def gmm(data, mode, max_iter = 100, x_tol = 0.01):
    # Find best Mixture Gaussian model
    last_loglike = float('-inf')
    mix = GaussianMixture(data, mode)
    best_mix = mix
    for i in range(max_iter):
        try:
            mix.iterate()
            if(last_loglike > mix.loglike):
                return best_mix
            if abs((last_loglike - mix.loglike)/mix.loglike) < x_tol:
                return mix
            last_loglike = mix.loglike
        except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
           pass
    return mix


def modes(data, max_iter = 100, x_tol = 0.01):
    # Find best Mixture Gaussian model
    mode = 2
    while True:
        try:
            best_mix = gmm(data, mode, max_iter, x_tol)
            modes = np.unique(np.array([sco.minimize(lambda x: -best_mix.pdf(x), [best_mix.dist[i].mu], method="CG", jac=lambda x: -best_mix.dpdf(x)).x for i in range(mode)]).flatten().round(2))
            if(modes.size < mode):
                break
            mode = mode + 1
        except (ZeroDivisionError, ValueError, RuntimeWarning):
            break
    return mode - 1 


