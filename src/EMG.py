# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:04:37 2019

@author: KavehB
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import seaborn as sns
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
    for i in range(max_iter):
        try:
            mix.iterate()
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

import pandas as pd    
x = np.linspace(start=-10, stop=10, num=1000)
df = pd.read_csv("../data/ph/1RXM_out.txt")
data = df.x
#
#print(modes(data))
#
#
#
#import time
#start = time.time() 
#mode = 2
#nmodes = 2
#best_mix = gmm(data, nmodes, max_iter = 200)
#z = best_mix.z
#for i in range(nmodes):
#    a = np.transpose(np.array([data, z[i]]))
#    sa = a[a[:,0].argsort()]
#    plt.plot(sa[:, 0], sa[:, 1], label=str(i + 1) + 'z')
#
#x1 = np.linspace(start=min(data), stop=max(data), num=1000)
#sns.distplot(data, bins=20, kde=False, norm_hist=True)
#dists = [[best_mix.mix[i]*best_mix.dist[i].pdf(e) for e in x1]  for i in range(nmodes)]
##all_dist = [gmmres.pdf(e) for e in x2]
#for i in range(nmodes):
#    plt.plot(x1, dists[i], label=str(i + 1))
##plt.plot(x1, all_dist, label='Inverse Gamma mixture')
#plt.legend()


#end = time.time()
#
#print(best_mix)
##
##print(end - start)
##
##print(len(data))
#
##print([len([z for z in x if z < 0.1]) for x in best_mix.z] )
#
#z = np.array(best_mix.z)
#print([sum(x)for x in z])
##
##print(best_mix)
##print([sum(z) for z in best_mix.z])
##print([np.mean(z) for z in best_mix.z])
##print([np.median(z) for z in best_mix.z])
##print([np.var(z) for z in best_mix.z])
##print(np.mean(data))
#
#
#
#
##
###mixture
# x = np.linspace(start=min(data), stop=max(data), num=1000)
# sns.distplot(data, bins=100, kde=False, norm_hist=True)
# dists = [[best_mix.mix[i]*best_mix.dist[i].pdf(e) for e in x]  for i in range(mode)]
# all_dist = [best_mix.pdf(e) for e in x]

# for i in range(mode):
#     plt.plot(x, dists[i], label=str(i + 1))

# plt.plot(x, all_dist, label='gaussian mixture')
# plt.legend()
##
#
#result = st.kstest(data, best_mix.cdf)
#print(result)
#
#
##print(st.kstest(data, best_mix.cdf))
#
#print([sco.minimize(lambda x: -best_mix.pdf(x), [best_mix.dist[i].mu], method="CG", jac=lambda x: -best_mix.dpdf(x)).x for i in range(mode)] )


