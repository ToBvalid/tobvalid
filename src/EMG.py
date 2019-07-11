# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:04:37 2019

@author: KavehB
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import scipy.stats as st
import seaborn as sns
from math import sqrt, log, exp, pi
import embase as embase
import pandas as pd

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

    def cdf(self, datum):
        return 0.5*(1 + erf((datum - self.mu)/(sqrt(2)*self.sigma)))
    #printing model values
    def __repr__(self):
        return 'Gaussian({0:4.6}, {1:4.6})'.format(self.mu, self.sigma)


class GaussianMixture(embase.Mixture):
    "Model mixture of N univariate Gaussians and their EM estimation"

    def __init__(self, data, n):
        super().__init__(data, n)
        self.dist = [Gaussian(0, 1) for i in range(self.mode)]
        self.log = [0]*self.mode
        for i in range(self.mode):
            self.dist[i].mu =  (2*i + 1)/(2*self.mode)
            self.dist[i].sigma = 1/(self.mode*sqrt(12))
    
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

    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"
        self.Mstep(self.Estep())  
        
    def __repr__(self):
        return ''.join(['GaussianMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['Mixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])



x = np.linspace(start=-10, stop=10, num=1000)
df = pd.read_csv("../data/ph/1F17_out.txt")
data = df.x


# Find best Mixture Gaussian model
n_iterations = 1000
n_random_restarts = 1
best_mix = None
best_loglike = float('-inf')
last_loglike = float('-inf')
print('Computing best model with random restarts...\n')
for _ in range(n_random_restarts):
    mix = GaussianMixture(data, 1)
    for i in range(n_iterations):
        try:
            mix.iterate()
            if mix.loglike > best_loglike:
                best_loglike = mix.loglike
                best_mix = mix

            if abs((last_loglike - mix.loglike)/mix.loglike) < 0.01:
                best_loglike = mix.loglike
                best_mix = mix
                break
            last_loglike = mix.loglike
        except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
           pass
print(best_loglike)
print(best_mix )
print('\n\nDone. 🙂')



#mixture
x = np.linspace(start=min(data), stop=max(data), num=1000)
sns.distplot(data, bins=20, kde=False, norm_hist=True)
g_both = [best_mix.pdf(e) for e in x]
plt.plot(x, g_both, label='gaussian mixture')
plt.legend()

histo, bin_edges = np.histogram(data, bins='auto', normed=False)
number_of_bins = len(bin_edges) - 1
observed_values = histo
cdf = best_mix.cdf(bin_edges)
expected_values = len(data) * np.diff(cdf)
c , p = st.chisquare(observed_values, expected_values)
print(c)
print(p)

