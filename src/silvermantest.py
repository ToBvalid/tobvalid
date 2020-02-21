# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:21:25 2019

@author: KavehB
"""

import tobevalid.stats.silverman as sv
import gparser as gp
import tobevalid.stats.pheight as ph
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tobevalid.mixture.gaussian_mixture import GaussianMixture



def silverman(file, data, s, neighbour = 1):

    p_data = ph.peak_height(data, s)
    
    d = p_data
    mode, kernel = sv.silverman(d, neighbour = neighbour)
    
    
    plt.figure(figsize=(12.8, 4.8))
    sns.distplot(d, bins=100, kde=False, hist_kws=dict(edgecolor="k", linewidth=2), norm_hist=True)
    p_x = np.linspace(start=min(d), stop=max(d), num=1000)
    dvalues = kernel.pdf(p_x)
    plt.plot(p_x, dvalues, label="pdf")
    plt.title("Silvermann. Peak height. File: {}. Mode number {}. Modes {}".format(file, mode[0], mode[1]))
    

    return mode


    

file = "5TU8_out.pdb"
(s, data) = gp.gemmy_parse("../data/pdb/"+ file)
modes = silverman(file, data, s, neighbour=10)


mode = modes[0]
p_data = ph.peak_height(data, s)
mixture = GaussianMixture(mode)
mixture.fit(p_data)

plt.figure(figsize=(12.8, 4.8))
p_x = np.linspace(start=min(p_data), stop=max(p_data), num=1000)
sns.distplot(p_data, bins=100, kde=False, hist_kws=dict(edgecolor="k", linewidth=2), norm_hist=True)
dvalues = mixture.pdf(p_x)
mvalues = mixture.mixpdf(p_x)
for i in np.arange(mode):
    plt.plot(p_x, mvalues.T[i]*mixture.mix[i], label="mix " + str(i + 1))
plt.plot(p_x, dvalues, label="GMM")
plt.legend()
plt.title("GMM. Peak height. File: {}.".format(file)) 




from tobevalid.mixture.invgamma_mixture import InverseGammaMixture

inv = InverseGammaMixture(mode, tol = 1e-15)
inv.fit(data, z=mixture.Z)


plt.figure(figsize=(12.8, 4.8))
p_x = np.linspace(start=min(data), stop=max(data), num=1000)
sns.distplot(data, bins=100, kde=False, hist_kws=dict(edgecolor="k", linewidth=2), norm_hist=True)
dvalues = inv.pdf(p_x)
plt.plot(p_x, dvalues, label="IGMM")
plt.legend()  
mvalues = inv.mixpdf(p_x)
for i in np.arange(mode):
    plt.plot(p_x, mvalues.T[i]*inv.mix[i], label="mix " + str(i + 1))
    
plt.legend()    
plt.title("IGMM. B-Value. File: {}.".format(file)) 


#in_dir = "../data/pdb/"
#files = sv.get_files(in_dir)
#files.sort()
#files = list(files)
#ch = np.random.choice(np.arange(len(files)), size = 20, replace= False)
##interval = [41 , 60]
##i = 1
#for i in ch:
##    if i < interval[0]:
##        i = i + 1
##        continue
#    file = files[i]
#    (s, data) = gp.gemmy_parse("../data/pdb/"+ file)
#    
#    if len(data) < 1000 or s == 0:
#        continue
#    silverman(file, data, s, neighbour=3)
##    i = i + 1
##    if i > interval[1]:
##        break
       