# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:50:53 2019

@author: KavehB
"""

# In[1]:
import math as math
import embase as embase
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import special
from scipy.special import erf
import scipy.stats as st
import scipy as sp
import seaborn as sns
import EMG as gm
import EMIG as igm
import EMM as emm
import silverman as sv
sns.set_style("white")
import pheight as ph
from math import sqrt, log, exp, pi
from random import uniform
import bparser as bp
import pandas as pd
import gparser as gp



(s, data) = gp.gemmy_parse("../data/pdb/4BB7_out.pdb")
p_data = ph.peak_height(data, s)

#plt.subplot(4, 2, 1)
#sns.distplot(data, bins=100, kde=False, norm_hist=True)
#
#plt.subplot(4, 2, 2)
#sns.distplot(p_data, bins=100, kde=False, norm_hist=True)

#nmodes = sv.boot_silverman(p_data, length = 1000)[0]
#print(nmodes)
import time
start = time.time()
nmodes = 2
gmmres =  gm.emgmm(p_data, nmodes)

end = time.time()
print(end - start)
print(gmmres.mixture)
z = gmmres.z

#N = np.sum(z, axis =1)
#print(np.sum((z*p_data).T*np.reciprocal(N, dtype=float), axis =0))
#plt.figure()
#for i in range(nmodes):
#    a = np.transpose(np.array([p_data, z[i]]))
#    sa = a[a[:,0].argsort()]
#    plt.plot(sa[:, 0], sa[:, 1], label=str(i + 1) + 'z')
#
#p_x = np.linspace(start=min(p_data), stop=max(p_data), num=1000)
#sns.distplot(p_data, bins=100, kde=False, norm_hist=True)
#dists = [np.vectorize(gmmres.mixture.pdf) for d in gmmres.mixture.dist]
#dvalues = np.array([d(p_x) for d in dists])
#mvalues = np.sum(np.multiply(dvalues.T, gmmres.mixture.mix), axis = 1).T
#plt.plot(p_x, mvalues, label="PH Gaussian Mixture")
#for i in range(nmodes):
#    plt.plot(p_x, dvalues[i], label=str(i + 1))
#plt.legend()

#x = np.arange(1,11).reshape(2, 5)
#y = np.arange(2,12).reshape(2, 5)
#print(y.T + np.full(2, 1))
#best_gm = gm.emgmm(data, nmodes, max_iter = 30, x_tol = 0.000000000001)
#print(best_gm.mixture)   
#
#plt.figure()
#x2 = np.linspace(start=min(data), stop=max(data), num=1000)
#sns.distplot(data, bins=100, kde=False, norm_hist=True)
#dists = [[best_gm.mixture.mix[i]*best_gm.mixture.dist[i].pdf(e) for e in x2]  for i in range(nmodes)]
#all_dist = [best_gm.mixture.pdf(e) for e in x2]
#for i in range(nmodes):
#    plt.plot(x2, dists[i], label=str(i + 1))
#plt.plot(x2, all_dist, label='BF Gaussian mixture')
#plt.legend()



import EMIG as igm
start = time.time()
igmix = igm.igmm(data,  nmodes, gmmres.z[::-1], z_tol=0.1, max_iter=30,  x_tol=0.00000000001)
end = time.time()
print(igmix)
print(end - start)

plt.figure()
x3 = np.linspace(start=min(data), stop=max(data), num=1000)
sns.distplot(data, bins=100, kde=False, norm_hist=True)
iall_dist = [igmix.pdf(e) for e in x3]
plt.plot(x3, iall_dist, label='Inverse Gamma mixture')         
#x = np.arange(10).reshape(2,5)
#y = np.arange(1, 11).reshape(2,5)   
#
#print(np.sum(x*np.log(y), 1))
