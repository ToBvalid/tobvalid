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
    u = (datum - mu) / np.abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * np.abs(sigma))) * np.exp(-u * u / 2)
    return y

   
def emgmm(d, mode, max_iter = 100, x_tol = 0.001):
    # Initialization
    data = np.array(d)
    datan = np.repeat(data[np.newaxis,...], mode, axis=0).T
    loglike = 0.
    mix = np.array([1./mode]*mode)
    dist = None
    mu_min = np.min(data)
    mu_max = np.max(data)
    mu = np.array([mu_min +  (mu_max - mu_min)*(2.*i + 1)/(2.*mode) for i in range(mode)])
    sigma =   np.array([(mu_max-mu_min)/(mode*np.sqrt(12.)) for i in range(mode)])
    
    
    last_loglike = float('-inf')
    dist = np.vectorize(pdf)
    
    best_mix =   GMMResult({'loglike':loglike, "mixture":GaussianMixture(mu, sigma, mix), "nit":1, "success":False, "z":None})
    for i in np.arange(max_iter):
        
        #E Step
        
        values =  dist(datan, mu, sigma).T
        wp = values.T*mix
        den = np.sum(wp, axis = 1)
        z = wp.T*np.reciprocal(den, dtype=float)
       
        #M Step
        N = np.sum(z, axis = 1)
        
        mu = np.sum((z* d).T *np.reciprocal(N, dtype=float), axis = 0 )
       
        diff = datan  - mu  
        diffSquare = diff*diff
        wdiffSquare = z.T*diffSquare
        sigma = np.sqrt(np.sum(wdiffSquare*np.reciprocal(N, dtype=float), axis = 0 ))
        mix = N*(1./np.sum(N))
        

        loglike = -np.sum(np.log(wp[wp > 1e-07]))
        cur_mix =   GMMResult({'loglike':loglike, "mixture":GaussianMixture(mu, sigma, mix), "nit":i+1, "success":True, "z":z})

        if (sigma < ((mu_max - mu_min)/100)).any():
            return best_mix    
        if np.abs((last_loglike - loglike)/loglike) < x_tol:
            return cur_mix
        best_mix = cur_mix
        last_loglike = loglike

   
    return GMMResult({'loglike':loglike, "mixture":GaussianMixture(mu, sigma, mix), "nit":max_iter, "success":True, "z":z})

def gmm_modes(d, max_iter = 100, x_tol = 0.001, mod_tol = 0.2, mix_tol = 0.1, peak_tol = 0.9, ret_mix = False):
#    lastres = emgmm(d, 1, max_iter, x_tol)
    lastres = None
    mode = 2
    while (True):
        gmmres = emgmm(d, mode, max_iter, x_tol)
        z = gmmres.z
         
        if ((gmmres.mixture.mix < mix_tol).any()):
            break
        if(min(z.sum(axis = 1)*gmmres.mixture.mix/d.size) < 0.1):
            break
        it = 0
        for i in np.arange(mode - 1):
            z1 = z[i].sum()
            z2 = z[i + 1].sum()
            z12 = z[i, z[i] <= z[i + 1]].sum() + z[i + 1, z[i] >= z[i + 1]].sum()
            if (2*z12/(z1 + z2) >= mod_tol):
                break
            peak1 = gmmres.mixture.mix[i]/(np.sqrt(2*np.pi)*gmmres.mixture.dist[i].sigma)
            peak2 = gmmres.mixture.mix[i + 1]/(np.sqrt(2*np.pi)*gmmres.mixture.dist[i + 1].sigma)
            if(abs(gmmres.mixture.dist[i + 1].mu - gmmres.mixture.dist[i].mu) < 1*(gmmres.mixture.dist[i + 1].sigma + gmmres.mixture.dist[i].sigma) 
                and min(peak1, peak2)/max(peak1, peak2) > peak_tol):
                break
            it += 1
        if(it < mode - 1):
            break
        lastres = gmmres
        mode += 1
    if(ret_mix == False):
        return mode - 1
    if( mode == 2):
        return (mode - 1, emgmm(d, 1, max_iter, x_tol))
    return (mode - 1, lastres)