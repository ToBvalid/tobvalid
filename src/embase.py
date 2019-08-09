# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:07:34 2019

@author: KavehB
"""
import numpy as np

class Distribution:
    def pdf(self, x):
        pass
    def cdf(self, x):
        pass
    def __repr__(self):
        pass
    def size(self):
        pass
    def setParams(self, params):
        pass
    def getParams(self):
        pass

class Mixture:
    def __init__(self, data, n):
        self.mode = n
        self.data = np.array(data)
        self.loglike = 0.
        self.mix = np.array([1/self.mode]*self.mode)

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
    
    def dpdf(self, x):
        return sum([w*d.dpdf(x) for (w, d) in zip (self.mix, self.dist)])
    
    def cdf(self, x):
        return sum([w*d.cdf(x) for (w, d) in zip (self.mix, self.dist)])

    def __repr__(self):
        pass

    def __str__(self):
        pass