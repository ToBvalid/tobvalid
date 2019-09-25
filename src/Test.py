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
import time
np.__version__

#(s, data) = gp.gemmy_parse("../data/pdb/1OB1_out.pdb")
#p_data = ph.peak_height(data, s)

#df = pd.read_csv("../data/ph/5EED_out.txt")
#p_data = df.x.values
#print("Length: {}".format(str(len(p_data))))
#
#plt.figure()
#sns.distplot(data, bins=30, kde=False, norm_hist=True)
#
#plt.figure()
#sns.distplot(p_data, bins=30, kde=False, norm_hist=True)
#
#    
#start = time.time()
# 
#res =  gm.gmm_modes(p_data, ret_mix=True, max_iter=50)
#nmodes = res[0]
#gmmres =  res[1]
#end = time.time()
#print("time: {}".format(end - start))
#print(gmmres.mixture)
#print(gmmres.nit)
#z = gmmres.z
#
#N = np.sum(z, axis =1)
#print(np.sum((z*p_data).T*np.reciprocal(N, dtype=float), axis =0))
#plt.figure()
#for i in range(nmodes):
#    a = np.transpose(np.array([p_data, z[i]]))
#    sa = a[a[:,0].argsort()]
#    plt.plot(sa[:, 0], sa[:, 1], label=str(i + 1) + 'z')
#
#
#p_x = np.linspace(start=min(p_data), stop=max(p_data), num=1000)
#sns.distplot(p_data, bins=30, kde=False, norm_hist=True)
#dists = [np.vectorize(gmmres.mixture.pdf) for d in gmmres.mixture.dist]
#dvalues = np.array([d(p_x) for d in dists])
#mvalues = np.sum(np.multiply(dvalues.T, gmmres.mixture.mix), axis = 1).T
#
#plt.plot(p_x, mvalues, label="PH Gaussian Mixture")
#
#for i in range(nmodes):
#    plt.plot(p_x, dvalues[i], label=str(i + 1))
#plt.legend()


in_dir = "../data/pdb"
files = sv.get_files(in_dir)
for file in files:
    (s, data) = gp.gemmy_parse(in_dir + "/" + file)
    if(len(data) < 1000):
        continue
    p_data = ph.peak_height(data, s)
    start = time.time()
 
    res =  gm.gmm_modes(p_data, ret_mix=True)
    nmodes = res[0]
#    end = time.time()
#    print("file {}, time: {} Length: {} modes {}".format(file, end - start, len(p_data), nmodes))
    gmmres =  res[1]
    end = time.time()
    print("file {}, time: {} iter {} Length: {} modes {}".format(file, end - start, gmmres.nit, len(p_data), nmodes))
#    print("file {}, modes {}".format(file, str(nmodes)))
    z = gmmres.z
    print(gmmres.mixture)
    N = np.sum(z, axis =1)
    plt.figure()
    for i in range(nmodes):
        a = np.transpose(np.array([p_data, z[i]]))
        sa = a[a[:,0].argsort()]
        plt.plot(sa[:, 0], sa[:, 1], label=str(i + 1) + 'z')
    
    
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
