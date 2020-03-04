# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:44:05 2019

@author: KavehB
"""



import numpy as np
from scipy import stats

  



#This function double value and list of doubles. Returns true if value contains in the list
def inlist(x, l, x_tol = 1e-5):
    for v in l:
        if np.abs(x - v) < x_tol:
            return True
    return False    

 #This function remove duplicates in list and sorts its values
def remove_dup(duplicate, x_tol = 1e-5): 
    final_list = [] 
    for num in duplicate: 
        if not inlist(num, final_list, x_tol): 
            final_list.append(num) 
    return final_list  


def filter_bymax(l, f):
    mx = max([f(v) for v in l])
    return [v for v in l if f(v)/mx > 0.1]

  

def modes(data, kernel, x_tol = 1e-5, rnd = 2, neighbour = 1):    
    result = list()
    length = np.max(data) - np.min(data)
    line = np.linspace(start= np.min(data) - 0.1*length, stop = np.max(data) + 0.1 * length, num = 100)
    f = kernel(line)  
    
    for i in np.arange(neighbour, len(line) - neighbour):
        is_max = True
        for j in np.arange(neighbour):
            if f[i] < f[i + j + 1] or f[i] < f[i - j - 1]:
                is_max = False
                break
        if is_max:
            result.append(np.round(line[i], rnd))  
            
    result.sort()        
    result = filter_bymax(remove_dup(result, x_tol), kernel)
    return (len(result), result) 


    


def silverman(data, rnd=2, neighbour = 10):
    kernel = stats.gaussian_kde(data, bw_method='silverman')
    result = modes(data, kernel, neighbour = neighbour, rnd = rnd)
    return (result, kernel)
           



