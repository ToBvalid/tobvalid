#%%
#from Mixture import *
#%%

# coding: utf-8

import math as math
from pathlib import Path
#For plotting
import matplotlib.pyplot as plt
#for matrix math
import numpy as np
#for normalization + probability density function computation
from scipy import stats
#for plotting
from scipy import special
from scipy.special import erf
import scipy.stats as st
import seaborn as sns
sns.set_style("white")


from math import sqrt, log, exp, pi
from random import uniform
#for data preprocessing
import pandas as pd

class Distribution:
    def pdf(self, x):
        pass
    def cdf(self, x):
        pass
    def __repr__(self):
        pass

class Mixture:
    def __init__(self, data, n):
        self.mode = n
        self.data = data
        self.loglike = 0.
        self.mix = [1/self.mode]*self.mode

        pass

    "Initialize distributuions"
    def Init(self, data):
        pass
    
    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        pass

    def Mstep(self, weights):
        "Perform an M(aximization)-step"
        pass
        
    def pdf(self, x):
        return sum([w*d.pdf(x) for (w, d) in zip (self.mix, self.dist)])
    
    def cdf(self, x):
        return sum([w*d.cdf(x) for (w, d) in zip (self.mix, self.dist)])

    def __repr__(self):
        pass

    def __str__(self):
        pass
#%%
def Peak_height(B, s):
    smax = 1/float(s)
    B = (B-np.amin(B)) + 0.01
    rho_zero = (8*np.pi/B)*(-smax*np.exp(-B*smax**2/4)+np.sqrt(np.pi/B)*erf(np.sqrt(B)*smax/2))
    rho_zero = rho_zero /max(rho_zero)
    return(rho_zero)
#%%
class Gaussian(Distribution):
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

# In[7]:

class GaussianMixture(Mixture):
    "Model mixture of N univariate Gaussians and their EM estimation"

    def __init__(self, data, n, sigma_min=.1, sigma_max=1):
        super().__init__(data, n)
        mu_min=min(data)
        mu_max=max(data)
        self.dist = [Gaussian(uniform(mu_min, mu_max), uniform(sigma_min, sigma_max)) for x in range(self.mode)]
    
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
    
            wp = [w/den for w in wp]; 
            # add into loglike
            self.loglike += log(sum(wp))
            # yield weight tuple
            yield wp

    def Mstep(self, weights):
        "Perform an M(aximization)-step"
        # compute denominators

        wps = list(zip(* weights))
        for i in range(self.mode):
            den = sum(wps[i])
            self.dist[i].mu =  sum(w * d / den for (w, d) in zip(wps[i], self.data))
            self.dist[i].sigma = sqrt(sum(w * ((d - self.dist[i].mu) ** 2)
                                for (w, d) in zip(wps[i], self.data)) / den)
            self.mix[i] = den / len(self.data)

       

    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"
        self.Mstep(self.Estep())  
        
    def __repr__(self):
        return ''.join(['GaussianMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['Mixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
# %%
#Return evenly spaced numbers over a specified interval.
x = np.linspace(start=-10, stop=10, num=1000)
#A normal continuous random variable.
#The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
y = stats.norm.pdf(x, loc=0, scale=1.5) 
#plot it!
plt.plot(x, y)
# In[3]:

#read our dataset\
df = pd.read_csv("1AON_out.csv")
#show first 5 examples (in BTC)
df.head(n=5)

# %%
data = df.x
# %%
data = Peak_height(data, 3)
# %%

#plot histogram
sns.distplot(data, bins=20, kde=False)

# In[9]:

# Find best Mixture Gaussian model
n_iterations = 100
n_random_restarts = 1
best_mix = None
best_loglike = float('-inf')
print('Computing best model with random restarts...\n')
for _ in range(n_random_restarts):
    mix = GaussianMixture(data, 2)
    for _ in range(n_iterations):
        try:
            mix.iterate()
            if mix.loglike > best_loglike:
                best_loglike = mix.loglike
                best_mix = mix
        except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
           pass
print(best_loglike)
print(best_mix )
print('\n\nDone. 🙂')


# In[10]:

#mixture
x = np.linspace(start=min(data), stop=max(data), num=1000)
sns.distplot(data, bins=20, kde=False, norm_hist=True)
g_both = [best_mix.pdf(e) for e in x]
plt.plot(x, g_both, label='gaussian mixture')
plt.legend()


#%%
histo, bin_edges = np.histogram(data, bins='auto', normed=False)
number_of_bins = len(bin_edges) - 1
observed_values = histo
cdf = best_mix.cdf(bin_edges)
expected_values = len(data) * np.diff(cdf)
c , p = st.chisquare(observed_values, expected_values)
print(c)
print(p)

