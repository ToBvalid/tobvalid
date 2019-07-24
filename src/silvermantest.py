# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:21:25 2019

@author: KavehB
"""

import silverman as sv
import pandas as pd
import time





#in_dir = "../data/ph"
#files = sv.get_files(in_dir)
#stats = {}
#for file in files:
#      print(file)
#      df = pd.read_csv(in_dir + "/" + file)
#      data = df.x
#      start = time.time()
#      res = sv.boot_silverman(data)
#      end = time.time()
#      stats[file] = (end - start, res)
#      
#print(stats)  

#in_dir = "../data/bfactor"
#files = sv.get_files(in_dir)
#stats = {}
#for file in files:
#      print(file)
#      df = pd.read_csv(in_dir + "/" + file)
#      data = df.to_numpy().flatten()
#      start = time.time()
#      res = sv.boot_silverman(data, length=2000, max_bandwidth = 10)
#      end = time.time()
#      stats[file] = (end - start, res)
#      
#print(stats)  


#
#[(key, stats[key][1][1][0], myfit[key][1][1][0]) for key in myfit.keys() if stats[key][1][1][0] != myfit[key][1][1][0]]
#
#
#import seaborn as sns
#import matplotlib.pyplot as plt
#import numpy as np
#
#key = '5HTF_out.txt'
#df = pd.read_csv(in_dir + "/" + key)
#data = df.x
#h = stats[key][1][0]
#f = sv.kde(data, h)
#x = np.linspace(start=min(data), stop=max(data), num=100)
#sns.distplot(data, bins=20, kde=False, norm_hist=True)
#g_both = [f(e) for e in x]
#plt.plot(x, g_both, label='gaussian mixture')
#plt.legend()
