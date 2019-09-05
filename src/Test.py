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

plt.figure()
for i in range(nmodes):
    a = np.transpose(np.array([p_data, z[i]]))
    sa = a[a[:,0].argsort()]
    plt.plot(sa[:, 0], sa[:, 1], label=str(i + 1) + 'z')

p_x = np.linspace(start=min(p_data), stop=max(p_data), num=1000)
sns.distplot(p_data, bins=100, kde=False, norm_hist=True)
dists = [np.vectorize(gmmres.mixture.pdf) for d in gmmres.mixture.dist]
dvalues = np.array([d(p_x) for d in dists])
mvalues = np.sum(np.multiply(dvalues.T, gmmres.mixture.mix), axis = 1).T
plt.plot(p_x, mvalues, label="PH Gaussian Mixture")
for i in range(nmodes):
    plt.plot(p_x, dvalues[i], label=str(i + 1))
plt.legend()


#best_gm = gm.gmm(data, nmodes, max_iter = 30, x_tol = 0.000000000001)
#print(best_gm)   

#plt.subplot(4, 2, 4)
#x2 = np.linspace(start=min(data), stop=max(data), num=1000)
#sns.distplot(data, bins=100, kde=False, norm_hist=True)
#dists = [[best_gm.mix[i]*best_gm.dist[i].pdf(e) for e in x2]  for i in range(nmodes)]
#all_dist = [best_gm.pdf(e) for e in x2]
#for i in range(nmodes):
#    plt.plot(x2, dists[i], label=str(i + 1))
#plt.plot(x2, all_dist, label='BF Gaussian mixture')
#plt.legend()


#best_gm.dist[0].mu
#best_gm.dist[1].mu
#best_gm.dist[0].sigma**2
#best_gm.dist[1].sigma**2
#
#best_gm.dist[0].mu - 3*best_gm.dist[0].sigma
#best_gm.dist[1].mu - 3*best_gm.dist[1].sigma


#def calcInit(gm):
#    return np.transpose(np.array([ [3.5, 2.5*3*d.sigma, d.mu - 3*d.sigma] for d in gm.dist]))

import EMIG as igm
igmix = igm.igmm(data,  nmodes, gmmres.z[::-1], z_tol=0.1, max_iter=30,  x_tol=0.000000000001, step = 1, fisher = True)
print(igmix)

plt.figure()
x3 = np.linspace(start=min(data), stop=max(data), num=1000)
sns.distplot(data, bins=100, kde=False, norm_hist=True)
iall_dist = [igmix.pdf(e) for e in x3]
plt.plot(x3, iall_dist, label='Inverse Gamma mixture')         

   
#igmix.logLike(1, 4.24208, 182.301, 88.9268) 
#igmix.dist[0].shift
#igmix.dist[1].shift
#
#igmix.dist[0].betta
#igmix.dist[0].betta/(igmix.dist[0].alpha - 1) + igmix.dist[0].shift
#
#igmix.dist[1].betta/(igmix.dist[1].alpha - 1) + igmix.dist[1].shift
#
#(igmix.dist[0].betta**2)/(((igmix.dist[0].alpha - 1)**2)*(igmix.dist[0].alpha - 2))
#
#(igmix.dist[1].betta**2)/(((igmix.dist[1].alpha - 1)**2)*(igmix.dist[1].alpha - 2))
#
#igmix.dist[0].betta*(igmix.dist[0].alpha - 1)
#igmix.dist[1].betta*(igmix.dist[1].alpha - 1)
#
#(igmix.dist[0].betta/(igmix.dist[0].alpha - 1)/3)**2
#(igmix.dist[1].betta/(igmix.dist[1].alpha - 1)/3)**2
#
#
#np.var(data)

#import EMM as emm
#emix = emm.emm(data, ["invgamma", "norm"],  gmmres.z[::-1], z_tol=0.01, max_iter=30,  x_tol=0.000000000001, step = 1)
#print(emix)         
#        
#plt.subplot(2, 1, 2)
#x4 = np.linspace(start=min(data), stop=max(data), num=1000)
#sns.distplot(data, bins=100, kde=False, norm_hist=True)
#
#eall_dist = [emix.pdf(e) for e in x4]
#plt.plot(x4, eall_dist, label='Inverse Gamma mixture')

#
#
#z = best_gm.z
#z = gmmres.z
#z.reverse()
#
#best_mix= igmm(data,  nmodes,  z, z_tol=0.5, max_iter=30, x_tol=0.000000000001, step = 1, fisher = False)
#print(best_mix)
#
#
#plt.subplot(3, 1, 3)
#x3 = np.linspace(start=min(data), stop=max(data), num=1000)
#sns.distplot(data, bins=100, kde=False, norm_hist=True)
#idists = [[best_mix.mix[i]*best_mix.dist[i].pdf(e) for e in x3]  for i in range(nmodes)]
#iall_dist = [best_mix.pdf(e) for e in x3]
#for i in range(nmodes):
#    plt.plot(x3, idists[i], label=str(i + 1))
#plt.plot(x3, iall_dist, label='Inverse Gamma mixture')
#plt.legend()
#
#
#print(best_mix.initShift)
#



#print(best_mix.initAlpha)
#
#
#
#zij = [[d.pdf(x) for x in data] for d in  best_gm.dist]
#sample_data = [np.random.choice(data, size = len(data) , p = [x/sum(z) for x in z]) for z in zij]
##sample_data = [np.random.normal(d.mu, d.sigma, size=int(len(data)) ) for (m, d) in  zip(best_gm.mix, best_gm.dist)]
#sns.distplot(sample_data[0], bins=100, kde=False, norm_hist=True)
#
#
#
#
#
#import sys
#
#def fit_to_all_distributions(data):
#    dist_names = ['norm', 'invgamma']
#
#    params = {}
#    for dist_name in dist_names:
#        try:
#            dist = getattr(st, dist_name)
#            param = dist.fit(data)
#
#            params[dist_name] = param
#        except Exception:
#            print("Error occurred in fitting")
#            params[dist_name] = "Error"
#
#    return params 
#
#
#def get_best_distribution_using_chisquared_test(data, params):
#
#    histo, bin_edges = np.histogram(data, bins='auto', normed=False)
#    number_of_bins = len(bin_edges) - 1
#    observed_values = histo
#
#    dist_names =  ['norm', 'invgamma']
#
#    dist_results = []
#
#    for dist_name in dist_names:
#
#        param = params[dist_name]
#        if (param != "Error"):
#            # Applying the SSE test
#            arg = param[:-2]
#            loc = param[-2]
#            scale = param[-1]
#            cdf = getattr(st, dist_name).cdf(bin_edges, loc=loc, scale=scale, *arg)
#            expected_values = len(data) * np.diff(cdf)
#            c , p = st.chisquare(observed_values, expected_values, ddof=number_of_bins-len(param))
#            dist_results.append([dist_name, c, p])
#
#
#    # select the best fitted distribution
#    best_dist, best_c, best_p = None, sys.maxsize, 0
#
#    for item in dist_results:
#        name = item[0]
#        c = item[1]
#        p = item[2]
#        if (not math.isnan(c)):
#            if (c < best_c):
#                best_c = c
#                best_dist = name
#                best_p = p
#
#    # print the name of the best fit and its p value
#
#    print("Best fitting distribution: " + str(best_dist))
#    print("Best c value: " + str(best_c))
#    print("Best p value: " + str(best_p))
#    print("Parameters for the best fit: " + str(params[best_dist]))
#
#    return best_dist, best_c, params[best_dist], dist_results
#
#
#params = [fit_to_all_distributions(d) for d in sample_data]
#print(params)
#test_result = [get_best_distribution_using_chisquared_test(d, p) for (d, p) in zip(sample_data, params)]
#
#print(np.transpose(np.array([p['invgamma'] for p in params])))
#
#param = [t[2] for t in test_result]
#dist = [getattr(st, t[0]) for t in test_result]
#                        
#
#plt.subplot(3, 2, 5)
#x3 = np.linspace(start=min(sample_data[1]), stop=max(sample_data[1]), num=1000)
#sns.distplot(sample_data[1], bins=100, kde=False, norm_hist=True)
#all_dist = [dist[1].pdf(e, *param[1][:-2], loc=param[1][-2], scale=param[1][-1]) for e in x3]
#plt.plot(x3, all_dist, label='BF Gaussian mixture')
#plt.legend()
#
#
##
##
##
#best_mix= [igm.igmm(d,  1,  z=[[1]*len(d)], z_tol=0.5, max_iter=2, x_tol=0.000000000001, step = 1, fisher = True) for d in sample_data]
#print(best_mix)
##
#
##plt.subplot(3, 2, 6)
#x4 = np.linspace(start=min(data), stop=max(data), num=1000)
#sns.distplot(data, bins=100, kde=False, norm_hist=True)
##iall_dist = [sum([best_gm.mix[i]*best_mix[i].pdf(e) for i in range(nmodes)]) for e in x3]
#iall_dist = [best_gm.mix[0]*best_mix[0].pdf(e) +  best_gm.mix[1]*best_mix[1].pdf(e)  for e in x4]
#plt.plot(x4, iall_dist, label='Inverse Gamma mixture')
#plt.legend()
