# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:44:05 2019

@author: KavehB
"""


from math import sqrt, log, exp, pi
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as sco
import scipy.integrate as integr
def kernel(datum, mu, h):
    u = (datum - mu) / abs(h)
    y = (1 / (sqrt(2 * pi) * abs(h))) * exp(-u * u / 2)
    return y

def approx(data, h):
    return lambda x: sum([kernel(x, d, h) for d in data])/(len(data))

def mapprox(data, h):
    return lambda x: sum([-1*kernel(x, d, h) for d in data])/(len(data))

def dapprox(data, h):
    return lambda x: sum([(-(x-d)/(h**2))*kernel(x, d, h) for d in data])/(len(data))

def ndapprox(data, h):
    return lambda x: -sum([(-(x-d)/(h**2))*kernel(x, d, h) for d in data])/(len(data))

def approxy(data, h):
    return lambda y, x: sum([(-(x-d)/(h**2))*kernel(x, d, h) for d in data])/(len(data))


def minimize(data, h, iter = 5, x_tol = 0.00001):    
    result = list()
    df = ndapprox(data, h)
    fm = mapprox(data, h)
    line = np.linspace(start=min(data), stop=max(data), num=iter)
    for t in line:
        res = sco.minimize(fm, [t], method="CG", jac=df)
        result.append(res.x[0])
    result.sort()        
    return remove_dup(result, x_tol)

def inlist(x, l, x_tol = 0.00001):
    for v in l:
        if abs(x - v) < x_tol:
            return True
    return False    

 
def remove_dup(duplicate, x_tol = 0.00001): 
    final_list = [] 
    for num in duplicate: 
        if not inlist(num, final_list, x_tol): 
            final_list.append(num) 
    return final_list     
    
df = pd.read_csv("../data/ph/5DA8_out.txt")
data = df.x
print(((4*(np.std(data)**5))/(3*len(data)))**(1/5))
print(1.06*np.std(data)*(len(data))**(-1/5))
h = 1.06*np.std(data)*(len(data))**(-1/5)

x = np.linspace(start=min(data), stop=max(data), num=100)
f = approx(data, h)
g_both = [f(e) for e in x]
plt.plot(x, g_both, label='silverman')
plt.legend()

print(minimize(data, h))

