# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:44:05 2019

@author: KavehB
"""


from math import sqrt, exp, pi
import pandas as pd
import numpy as np
import scipy.optimize as sco
import scipy.special as scp
import os
import itertools
from scipy import stats


def get_files(in_dir):
      in_directory = in_dir
      return list(itertools.chain(*[files for root, dirs, files in os.walk(in_directory) if  files]))
  

# Normal distribution Kernel
def kernel(datum, mu, h):
    u = (datum - mu) / abs(h)
    y = (1 / (sqrt(2 * pi) * abs(h))) * exp(-u * u / 2)
    return y


# This function accepts data and bandwidth and returns kernel density estimation function (kde)
def kde(data, h):
    return lambda x: sum([kernel(x, d, h) for d in data])/(len(data))

def fkde(x, data, h):
    return sum([kernel(x, d, h) for d in data])/(len(data))

# This function accepts data and bandwidth and returns kernel density estimation function (kde)
def ckde(data, h):
    return lambda x: sum([0.5*(1 + scp.erf((x - d)/(h*sqrt(2)))) for d in data])/(len(data))

# This function accepts data and bandwidth and returns derivative of kde. 
def dkde(data, h):
    return lambda x: sum([(-(x-d)/(h**2))*kernel(x, d, h) for d in data])/(len(data))

def fdkde(x, data, h):
    return sum([(-(x-d)/(h**2))*kernel(x, d, h) for d in data])/(len(data))


#This function double value and list of doubles. Returns true if value contains in the list
def inlist(x, l, x_tol = 0.00001):
    for v in l:
        if abs(x - v) < x_tol:
            return True
    return False    

 #This function remove duplicates in list and sorts its values
def remove_dup(duplicate, x_tol = 0.00001): 
    final_list = [] 
    for num in duplicate: 
        if not inlist(num, final_list, x_tol): 
            final_list.append(num) 
    return final_list  


#This function accepts data retruns bandwidth 
def get_width(data):
    return ((4*(np.std(data)**5))/(3*len(data)))**(1/5)

#This function accepts data retruns bandwidth using approximate formula 
def get_width_apr(data):
    return 1.06*np.std(data)*(len(data))**(-1/5)



def find_widths(in_dir):
    files = get_files(in_dir)
    result = {}
    for file in files:
          df = pd.read_csv(in_dir + "/" + file)
          data = df.x
          result[file] = get_width_apr(data)
    return result    

def filter_bymax(l, f):
    mx = max([f(v) for v in l])
    return [v for v in l if f(v)/mx > 0.1]

def filter_bymax2(l, f, data, h):
    mx = max([f(v, data, h) for v in l])
    return [v for v in l if f(v, data, h)/mx > 0.1]

"""
This function accepts data and bandwidth and returns list of modes
If bandwith equals to zero function calculates it
To find modes we use Conjugate Gradient method. 
Iter parameter defines number of iterations. 
Each iteration starts from different point.   
"""
def modes(data, h = 0, max_iter = 10, x_tol = 0.00001, rnd = 2):    
    result = list()
    if h == 0 :
        h = get_width_apr(data)
    df = dkde(data, h)
    fm = kde(data, h)
    start = min(data)
    end = max(data)
    delta = (end - start)/max_iter
    point = min(data)
    while point < end:
        res = sco.minimize(lambda x: -fm(x), [point], method="CG", jac=lambda x: -df(x))
        result.append(round(res.x[0], rnd))
        point = point + delta
    result.sort()        
    result = filter_bymax(remove_dup(result, x_tol), fm)
    return (len(result), result)


def modes2(data, h = 0, x_tol = 0.00001, rnd = 2):    
    result = list()
    if h == 0 :
        h = get_width_apr(data)
    df = dkde(data, h)
    f = kde(data, h)
    
    line = np.linspace(start=min(data), stop=max(data), num=100)
    min_point = list()
    s = df(min(data))
    for x in line:
        if(s*np.sign(df(x)) == -1):
            min_point.append(x)
        s = np.sign(df(x))
    i = 0    
    while i < len(min_point):    
#        res = sco.minimize(lambda x: -f(x), [min_point[i]], method="CG", jac=lambda x: -df(x))
#        result.append(round(res.x[0], rnd))
        result.append(round(min_point[i], rnd))
        i = i + 2
        
    result.sort()        
    result = filter_bymax(remove_dup(result, x_tol), f)
    return (len(result), result)   

def modes3(data, h = 0, x_tol = 0.00001, rnd = 2):    
    result = list()
    if h == 0 :
        h = get_width_apr(data)
    
    line = np.linspace(start=min(data), stop=max(data), num=100)
    min_point = list()
    s = fdkde(min(data), data, h)
    for x in line:
        if(s*np.sign(fdkde(x, data, h)) == -1):
            min_point.append(x)
        s = np.sign(fdkde(x, data, h))
    i = 0    
    while i < len(min_point):    
#        res = sco.minimize(lambda x: -f(x), [min_point[i]], method="CG", jac=lambda x: -df(x))
#        result.append(round(res.x[0], rnd))
        result.append(round(min_point[i], rnd))
        i = i + 2
        
    result.sort()        
    result = filter_bymax2(remove_dup(result, x_tol), fkde, data, h)
    return (len(result), result)  

class kdd(stats.rv_continuous):
    def __init__(self, data, h):
        super(kdd, self).__init__()
        self.data = data
        self.h = h
        self.pdff = kde(data, h)
        self.cdff = ckde(data, h)
    
    
    def _pdf(self, x):
        return self.pdff(x)

    def _cdf(self, x):
        return self.cdff(x)


def fit_data(data, cdf):
    histo, bin_edges = np.histogram(data, bins='auto', density= False)
    observed_values = histo
    cdf_v = [cdf(z) for z in bin_edges]
    expected_values = len(data) * np.diff(cdf_v)
    return stats.ks_2samp(observed_values, expected_values)


    
def silverman(data, x_tol = 0.00001, rnd=2, p_value = 0.05, min_bandwidth = 0, max_bandwidth = 1, fit = stats.kstest):
    min_b = min_bandwidth
    max_b = max_bandwidth
    h = (max_b + min_b)/2
    cdf = ckde(data, h)
    stat = fit(data, cdf)
    i = 0
    while (stat.pvalue < p_value or stat.pvalue > 2*p_value ):
       if (stat.pvalue < p_value):
           max_b = h
           h = (min_b + h)/2
       if (stat.pvalue > p_value):
           min_b = h
           h = (max_b + h)/2
       cdf = ckde(data, h)
       stat = fit(data, cdf)
       i = i + 1
    return (h, modes2(data, h = h), stat.pvalue, i)
           

def boot_silverman(data, x_tol = 0.00001, rnd=2, p_value = 0.05, min_bandwidth = 0, max_bandwidth = 1, fit = stats.kstest, samples=10, length = 1000):
    samp = np.random.choice(data, (samples, length), replace=True)
    result = [silverman(d, x_tol = x_tol, rnd = rnd,  p_value = p_value, min_bandwidth = min_bandwidth, max_bandwidth = max_bandwidth, fit = fit) for d in samp]
    nmode = int(round(np.mean([res[1][0] for res in result]), 0))
    modes = []
    for i in range(nmode):
        modes.append(round(np.mean([res[1][1][i] for res in result if res[1][0] == nmode]), rnd)) 
    return (nmode, modes)


#import seaborn as sns
#import matplotlib.pyplot as plt
#
#df = pd.read_csv("../data/ph/5HTF_out.txt")
#data = df.x
#x = np.linspace(stop=0, start=1, num=100)
#
#sns.distplot(data, bins=20, kde=False, norm_hist=True)


#import time
#
#start = time.time()
#res = boot_silverman(data) 
#end = time.time()
#print(end - start)
#print(res)
##for h in x:
#    res = modes2(data, h = h)
#    print(res[0])
#    if( res[0] > 3):
#        break
#
#    
#sampling = np.random.choice(data, (10, len(data)), replace=True)

#in_directory = "../data/ph"
#
#res= pick_width_in_directory(in_directory)
#
#print(res)

""" 
    Example 1     
"""

#df = pd.read_csv("../data/ph/3AZI_out.txt")
#data = df.x
#result = silverman(data, p_value=0.1, fit=fit_data) 
#print(result)
#h = result[0]
#print(h)
#cdf = ckde(data, h)
#f = kde(data, h)
#df = dkde(data, h)
#
#
#import seaborn as sns
#import matplotlib.pyplot as plt
#x = np.linspace(start=min(data), stop=max(data), num=100)
#sns.distplot(data, bins=20, kde=False, norm_hist=True)
#g_both = [f(e) for e in x]
#plt.plot(x, g_both, label='gaussian mixture')
#plt.legend()

#
#import time
#
#start = time.time()
#histo, bin_edges = np.histogram(data, bins='auto', density= False)
#number_of_bins = len(bin_edges) - 1
#observed_values = histo
#cdf_v = [cdf(z) for z in bin_edges]
#expected_values = len(data) * np.diff(cdf_v)
#c , p = stats.chisquare(observed_values, expected_values)
#end = time.time()
#print('kolmogorov')
#print(stats.ks_2samp(observed_values, expected_values))
#print(p)
#
#print(end - start)
""" 
    Example 2     
"""

#df = pd.read_csv("../data/ph/2CFY_out.txt")
#data = df.x
#
#h = get_width_apr(data)
#print(h)
#cdf = ckde(data, h)
#f = kde(data, h)
#df = dkde(data, h)
#
#
#print(modes2(data, h = h))
##print(modes(data, h = h))
#
##import seaborn as sns
##import matplotlib.pyplot as plt
##x = np.linspace(start=min(data), stop=max(data), num=100)
##sns.distplot(data, bins=20, kde=False, norm_hist=True)
##g_both = [f(e) for e in x]
##plt.plot(x, g_both, label='gaussian mixture')
##plt.legend()
#
#
#import time
#
#start = time.time()
#histo, bin_edges = np.histogram(data, bins='auto')
#print(histo)
#number_of_bins = len(bin_edges) - 1
#observed_values = histo
#cdf_v = [cdf(z) for z in bin_edges]
#print(cdf_v)
#expected_values = len(data) * np.diff(cdf_v)
#c , p = stats.chisquare(observed_values, expected_values)
#end = time.time()
#print(p)
#
#print(end - start)
#
#start = time.time()
#res = stats.kstest(data, cdf)
#end = time.time()
#
#print(res)
#print(end - start)

''' 
    Example 3 
    Finding modes of all data in directory     
'''

#result = {}
#in_dir = "../data/ph"
#files = get_files(in_dir)
#for file in files:
#      print(file)
#      df = pd.read_csv(in_dir + "/" + file)
#      data = df.x
#      result[file] = modes2(data)
#print(result)


'''
'''

#from scipy.stats import chisquare
#import statsmodels.api as sm
#from scipy import stats
#
#
#
#
#in_dir = "../data/ph"
#files = get_files(in_dir)
#for file in files:
#      df = pd.read_csv(in_dir + "/" + file)
#      data = df.x
#      h = get_width_apr(data)
#      cdf = ckde(data, h)
#     
#      print(file)
#      print(stats.kstest(data, cdf))


