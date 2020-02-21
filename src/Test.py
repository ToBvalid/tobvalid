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

import EMG as gm
import EMIG as igm
import silverman as sv
import seaborn as sns
sns.set_style("white")
import pheight as ph
from math import sqrt, log, exp, pi
from random import uniform
import bparser as bp
import pandas as pd
import gparser as gp
import time
from scipy.stats import norm

dist = stats.norm()


def analyze(p_data, file):
    start = time.time()
    res =  gm.gmm_modes(p_data, ret_mix=True)
    nmodes = res[0]

    gmmres =  res[1]
    end = time.time()
    print("file {}, time: {} iter {} Length: {} modes {}".format(file, end - start, gmmres.nit, len(p_data), nmodes))
    print(gmmres.mixture)
    plt.figure()
#    for i in range(nmodes):
#        a = np.transpose(np.array([p_data, z[i]]))
#        sa = a[a[:,0].argsort()]
#        plt.plot(sa[:, 0], sa[:, 1], label=str(i + 1) + 'z')
    
    
    p_x = np.linspace(start=min(p_data), stop=max(p_data), num=1000)
    sns.distplot(p_data, bins=30, kde=False, norm_hist=True)
    dists = [np.vectorize(gmmres.mixture.pdf) for d in gmmres.mixture.dist]
    dvalues = np.array([d(p_x) for d in dists])
    mvalues = np.sum(np.multiply(dvalues.T, gmmres.mixture.mix), axis = 1).T
    
    plt.plot(p_x, mvalues, label="PH Gaussian Mixture")
    
    for i in range(nmodes):
        plt.plot(p_x, dvalues[i], label=str(i + 1))
    plt.legend()
    plt.title(file)

    return res

def ful_analyze(data, p_data):    
    res =  gm.gmm_modes(p_data, ret_mix = True)
    
    nmodes= res[0]
    gmmres= res[1]
    
    igmix = igm.igmm(data, nmodes, gmmres.z)

    return igmix


def plot(result):
    for file in result:
        plt.figure()
        mixture = result[file].mixture
        data = mixture.data
        x = np.linspace(start=min(data), stop=max(data), num=1000)
        sns.distplot(data, bins=30, kde=False, norm_hist=True)
        dists = [np.vectorize(mixture.pdf) for d in mixture.dist]
        dvalues = np.array([d(x) for d in dists])
        mvalues = np.sum(np.multiply(dvalues.T, mixture.mix), axis = 1).T
        plt.plot(x, mvalues, label="Inverser Gamma Mixture")
        for i in range(len(mixture.mix)):
            plt.plot(x, dvalues[i], label=str(i + 1))
        plt.legend()
        plt.title(file + " " + "") 

def filterTheDict(dictObj, callback):
    newDict = dict()

    for (key, value) in dictObj.items():
        if callback((key, value)):
            newDict[key] = value
    return newDict




#in_dir = "../data/pdb/"
#files = sv.get_files(in_dir)
##files = ["2B1C_out.pdb", "2CC9_out.pdb", "2NXN_out.pdb", "3NVH_out.pdb", "4EST_out.pdb", "5E5Z_out.pdb"]
#result = {}
#for file in files:
#    (s, data) = gp.gemmy_parse(in_dir + file)
#    if(len(data) < 1000):
#        continue
#    if s == 0:
#        continue
#    p_data = ph.peak_height(data, s)
#    result[file] = ful_analyze(data, p_data)
#    
#success = filterTheDict(result, lambda elem: elem[1].success == True)
#fails = filterTheDict(result, lambda elem : elem[1].success == False)
#print()
#print()
#print()
#print()
#print()
#print("SUCCESSES .................................................")
#print()
#print()
#print()
#print()
#
#for file in success:
#    print(file, ":", success[file].mixture)    
#plot(success)
#print()
#print()
#print()
#print()
#print()
#print("FAILES .................................................")
#print()
#print()
#print()
#print()
#for file in fails:
#    print(file, ":", fails[file].mixture, " Reason: ", fails[file].Reason)
#    
#plot(fails)


    
file = "1A0C_out.pdb"
(s, data) = gp.gemmy_parse("../data/pdb/"+ file)
p_data = ph.peak_height(data, s)

plt.figure()
from scipy.stats import gaussian_kde
kernel = stats.gaussian_kde(p_data, bw_method='silverman')
sns.distplot(p_data, bins=30, kde=False, norm_hist=True)
p_x = np.linspace(start=min(p_data), stop=max(p_data), num=1000)
dvalues = kernel.pdf(p_x)
plt.plot(p_x, dvalues, label="pdf") 



#df = pd.read_csv("../data/ph/5EED_out.txt")
#p_data = df.x.values

#res = analyze(p_data, file)

from tobevalid.mixture.gaussian_mixture import GaussianMixture

#gmmres = gm.emgmm(p_data, 2)

#mixture = GaussianMixture(2)
#mixture.fit(p_data)
#result = GaussianMixture.search(p_data, ret_mix = True)
#
#nmodes= result[0]
#gmmres= result[1]
#
#print(nmodes)

sns.distplot(p_data, bins=30, kde=False, norm_hist=True)
for i in range(3):
#    nmodes = 2
    gmmres = GaussianMixture(i + 1)
    gmmres.fit(p_data)
    
    result = igm.igmm(data, i + 1, gmmres.Z.T, x_tol=0.0000000001)
    igmix = result.mixture
    plt.figure()
    p_x = np.linspace(start=min(data), stop=max(data), num=1000)
    sns.distplot(data, bins=30, kde=False, norm_hist=True)
    dvalues = igmix.pdf(p_x)
    plt.plot(p_x, dvalues, label="pdf")
    print(igmix.loglike)
print(igmix)          
#z = gmmres.Z
#
#result = igm.igmm(data, nmodes, gmmres.Z.T, x_tol=0.0000000001)
#igmix = result.mixture 
#
#
#
#print(igmix)
#plt.figure()
#p_x = np.linspace(start=min(data), stop=max(data), num=1000)
#sns.distplot(data, bins=30, kde=False, norm_hist=True)
#dvalues = igmix.pdf(p_x)
#plt.plot(p_x, dvalues, label="pdf") 
#
#
#
#plt.figure()
#i = 0
#for i in range(nmodes): 
#    p_x = np.linspace(start=min(data), stop=max(data), num=1000)
#    #sns.distplot(data, bins=30, kde=False, norm_hist=True)
#    sns.distplot(igmix.clusters()[i], bins=30, kde=False, norm_hist=True)
#    mvalues = igmix.dist[i].pdf(p_x)
#    plt.plot(p_x, mvalues, label="pdf")
##dists = [np.vectorize(d.pdf) for d in igmix.dist]
##dvalues = np.array([d(p_x) for d in dists])
##mvalues = np.sum(np.multiply(dvalues.T, igmix.mix), axis = 1).T
#dvalues = igmix.dist[i].pdf(p_x)
#plt.plot(p_x, mvalues, label="pdf")
#plt.title("1AON") 
#
#sns.distplot(igmix.clusters()[0], bins=30, kde=False, norm_hist=True)
#
#
#
#
#for i in range(igmix.mode):
#    plt.figure()
#    cl = igmix.clusters()[i]
#    cli = igm.igmm(cl, 1, np.array([np.ones(len(cl)).tolist()])   ) 
#    print(cli.mixture)
#    sns.distplot(cl, bins=30, kde=False, norm_hist=True)
#    p_x = np.linspace(start=min(data), stop=max(data), num=1000)
#    values = igmix.dist[i].pdf(p_x)
#    
#    plt.plot(p_x, values, label="pdf")
#
#
#
#result = []
#p_sum = 0
#for i in range(igmix.mode):
#    p_sum += igmix.dist[i].pdf(data).sum()
#    
#for i in range(igmix.mode): 
#    p = igmix.dist[i].pdf(data)
#    len_i = p.sum()/p_sum
#    result.append(np.random.choice(data.tolist(), int(len(igmix.data)*len_i), p=p/p.sum(), replace=False))
# 
#
#
#for i in range(igmix.mode):    
#    plt.figure()
#    
#    sns.distplot(result[i], bins=30, kde=False, norm_hist=True)
#    p_x = np.linspace(start=min(data), stop=max(data), num=1000)
#    values = igmix.dist[i].pdf(p_x)
#    
#    plt.plot(p_x, values, label="pdf")
#
#
#
#values = np.array([d.pdf(igmix.data) for d in igmix.dist])
#
#sums = np.matmul(igmix.mix, values)
#wps  = np.matmul(np.diag(igmix.mix), values)
#
#
#
#for i in range(len(wps)):
#    for j in range(len(wps[i])):
#        if wps[i][j] != 0:
#            wps[i][j] = wps[i][j]/sums[j]
#
#index = np.array(np.arange(igmix.mode))
#
#
#clusters = []
#
#for i in np.arange(len(index)):
#    clusters.append([])
#    
#for i in np.arange(len(data)):
#    idx = np.random.choice(index, 1, p = wps[:,i])[0]
#    clusters[idx].append(data[i])
#    
#    
#for i in range(igmix.mode):    
#    plt.figure()
#    
#    sns.distplot(result[i], bins=30, kde=False, norm_hist=True)
#    p_x = np.linspace(start=min(data), stop=max(data), num=1000)
#    values = igmix.dist[i].pdf(p_x)
#    
#    plt.plot(p_x, values, label="pdf")
#        
#        
#import ToBvalid as tov
#
#for i in range(igmix.mode):
#    cl = igmix.clusters()[i]
#    r = tov.estimateIGpars(cl)
#    print(r)
    
    
    

#import EMIGOLD as igmold
#igmix = igmold.igmm(data, nmodes, gmmres.z, max_iter=1)

#    
#df = pd.read_csv("../data/ph/5EED_out.txt")
#p_data = df.x.values


#
#res = analyze(p_data, file)
#analyze(dist.ppf(0.9*p_data), file + " 0.9 qnorm")

#in_dir = "../data/pdb"
#files = sv.get_files(in_dir)
##files = ["2B1C_out.pdb", "2CC9_out.pdb", "2NXN_out.pdb", "3NVH_out.pdb", "4EST_out.pdb", "5E5Z_out.pdb"]
#for file in files:
#    (s, data) = gp.gemmy_parse(in_dir + "/" + file)
##    if(len(data) > 3000  or len(data) < 1000 ):
##        continue
#    if s == 0:
#        s = 2 
#    p_data = ph.peak_height(data, s)
#    
##    analyze(p_data, file)
#    analyze(dist.ppf(p_data*0.9), file+ " 0.9 qnorm") 



