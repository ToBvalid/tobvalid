# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:35:03 2019

@author: KavehB
"""

import embase as embase
import EMG as gm
import EMIG as igm
import numpy as np
import math
from scipy import special
import itertools

class MixedMixture(embase.Mixture):
    "Model mixture of N univariate Inverse Gamma Distributions and their EM estimation"

    def __init__(self, data, names, z, c = 0.1, step = 1, fisher = True, mix=None):
        self.al0 = 3.5
        self.sig = 0.1
        self.epsilon = 1.0e-16
        self.names = names
        super().__init__(data, len(names))
        self.step = step
        self.fisher = fisher
        self.c = c
        self.z = z
        self.dist = []
        self.sizes = []
        
        if mix is not None:
            self.mix = mix
#            
        self.init()
    
        self.fvalue = None
        
        self.grad = None                                                                                                               
        
        self.d2l = np.zeros((sum(self.sizes), sum(self.sizes)))
            
        
        self.loglike = 0
        
    def preInit(self):
        N = [sum(x) for x in self.z]
        mdB = [sum([self.z[i][j]*self.data[j] for j in range(len(self.data)) ])/N[i] for i in range(self.mode)]    
        minB = [min([d for (d, zj) in zip(self.data, self.z[i]) if zj > self.c]) for i in range(self.mode)]
        self.initShift = minB
        self.initAlpha = [3.5]*self.mode
        self.initBetta = [(bm - b)*(a - 1) for (bm, b, a) in zip(mdB, self.initShift, self.initAlpha)]

    def init(self):
        self.preInit()
        i = 0
        for name in self.names:
            if name == "invgamma":
                self.initIGM(i)
            elif name == "norm":
                self.initGM(i)
            else:
                 raise Exception(name + ' distribution is not supported')
            self.sizes.append(self.dist[i].size())
            i = i + 1
            
    def initIGM(self, i):
        self.dist.append(igm.InverseGammaDistribution(self.initAlpha[i], self.initBetta[i], self.initShift[i]))
        
    def initGM(self, i):
        self.dist.append(gm.Gaussian(self.initShift[i] , math.sqrt(np.var(self.data))))           
    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        
        pdfs = [np.vectorize(d.pdf) for d in self.dist]
        values = np.array([f(self.data) for f in pdfs])  
        sums = self.mix@values
        wps = np.diag(self.mix)@values
   
        for i in range(len(wps)):
            for j in range(len(wps[i])):
                if wps[i][j] != 0:
                    wps[i][j] = wps[i][j]/sums[j]
            
        self.weights = wps  
        self.N = np.sum(self.weights, axis = 1)
        
        self.mix = self.N*(1/sum(self.N))

    def Mstep(self):
            "Perform an M(aximization)-step"            
            
            self.CalcFisherMatrix()
            ss = self.shiftcalc(self.epsilon)
            step = self.step
            iparams = np.array(list(itertools.chain(*[d.getParams() for d in self.dist])))
            ifvalue = self.loglike
            conv = False
            
            while not conv:
                border = 0
                for i in range(self.mode):
             
                    start = border
                    end = border + self.sizes[i]
#                    self.dist[i].setParams(iparams[start:end] + step*ss[start:end])    
                    if(self.names[i] == "invgamma"):
                        self.dist[i].setParams(iparams[start:end] + step*ss[start:end])
                    elif self.names[i] == "norm":
                        self.dist[i].mu =  sum(w * d / self.N[i] for (w, d) in zip(self.z[i], self.data))
                        self.dist[i].sigma = math.sqrt(sum(w * ((d - self.dist[i].mu) ** 2) for (w, d) in zip(self.z[i], self.data)) / self.N[i])               
                    
               
                self.CalcFisherMatrix()
    
                if self.loglike > ifvalue:
                    step = step*0.7
                else:
                    conv = True
               
                if step <= self.epsilon: 
                    if not conv:
                        border = 0
                        for i in range(self.mode):
                             start = border
                             end = border + self.sizes[i]
                             self.dist[i].setParams(iparams[start:end])
                             border = border + self.sizes[i]
                        self.loglike = ifvalue
                    break

            return conv
    
    def CalcFisherMatrixIG(self, i):
        w = 1/self.sig**2 
        
  
        d = self.dist[i]    
        n = self.N[i]
        zj = self.weights[i]    
        delta = [(x - d.shift, z) for (x, z) in zip(self.data, zj)  if(x > d.shift)]
        
        weightedLogDelta = sum([z*math.log(x) for (x, z) in delta])
        weightedInverseDelta = sum([z/x for (x, z) in delta])
        inverseDelta = sum([z/x for (x, z)  in delta])
        inverseSquareDelta = sum([z/(x**2) for (x, z)  in delta])
        
       
        fvalue = - n*d.alpha*math.log(d.betta) + n*math.lgamma(d.alpha) + d.betta*weightedInverseDelta + (d.alpha + 1)*weightedLogDelta  + w*(d.alpha - self.al0)**2/2 
        gradAlpha = n*special.psi(d.alpha) - n*math.log(d.betta) + weightedLogDelta  + w*(d.alpha - self.al0)
        gradBetta = -n*d.alpha/d.betta + inverseDelta   
        gradB = d.betta*inverseSquareDelta - (d.alpha + 1)*inverseDelta                                                                                                                         
    
        d2f =  np.array([[n*special.polygamma(1, d.alpha) + w, -n/d.betta, -n*d.alpha/d.betta ],
                 [-n/d.betta, n*d.alpha/d.betta**2, n*d.alpha*(d.alpha + 1)/d.betta**2],
              [-n*d.alpha/d.betta,  n*d.alpha*(d.alpha + 1)/d.betta**2,  n*d.alpha*(d.alpha + 1)*(d.alpha + 3)/d.betta**2]])
        
        return (self.mix[i]*fvalue, [self.mix[i]*gradAlpha, self.mix[i]*gradBetta, self.mix[i]*gradB], self.mix[i]*d2f)        
                    
    def CalcFisherMatrixG(self, i):
        
  
        d = self.dist[i]    
        n = self.N[i]
        zj = self.weights[i] 
        
        fvalue = 1*sum((w * ((datum - d.mu) ** 2))/(d.sigma**2) - w*math.log(d.sigma**2) for (w, datum) in zip(zj , self.data))/2
        
        diff = self.data - np.array([d.mu]*len(self.data))
        wdiff = np.multiply(diff, zj)
        wdiffSquare = np.multiply(np.dot(diff, diff), zj)
        
        gradMu = -np.sum((1/d.sigma**2)*wdiff)
        gradSigma = n/d.sigma - np.sum((1/d.sigma**3)*wdiffSquare)                                                                                                                            
    
        d2f =  np.array([[1/(d.sigma**2), 0 ], [0, 1/(d.sigma**4)]])
        return (self.mix[i]*fvalue, [self.mix[i]*gradMu, self.mix[i]*gradSigma], self.mix[i]*d2f) 
    
    def CalcFisherMatrix(self):

        self.d2l = np.zeros((sum(self.sizes), sum(self.sizes)))
        
        derivatives = []
        for i in range(self.mode):   
             
             if self.names[i] == "invgamma":
                 derivatives.append(self.CalcFisherMatrixIG(i))
             elif self.names[i] == "norm":
                 derivatives.append(self.CalcFisherMatrixG(i))
        
        self.loglike = np.array([dev[0] for dev in derivatives]).sum()
        
        self.grad = np.array(list(itertools.chain(*[dev[1] for dev in derivatives])))
        d2f = [dev[2] for dev in derivatives]
        
        border = 0
        for i in range(self.mode):
            for j in range(self.sizes[i]):
                for k in range(self.sizes[i]):
                    self.d2l[border + j, k + border] =  d2f[i][j][k]
            border = border + self.sizes[i]
            
    def shiftcalc(self, tol = 1.0e-4):     
        inv = np.linalg.inv(self.d2l + tol*np.random.randn(sum(self.sizes), sum(self.sizes))) 
        shiftc = np.matmul(-inv, np.transpose(self.grad))
        return shiftc
    
            
    def iterate(self, N=1, verbose=False):
        self.Estep()
        return self.Mstep()
        
    def __repr__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
def emm(data, names, z, z_tol = 0.1, step = 1, fisher = True, mix=None, max_iter = 100, x_tol = 0.0000001):
    last_loglike = float('inf')
    mix = MixedMixture(data, names, z, c = z_tol, step = step, fisher = fisher, mix = mix)
    best_mix = mix
    for i in range(max_iter):
        if not mix.iterate():
            return best_mix
        if(last_loglike < mix.loglike):
            best_mix = mix
        if abs((last_loglike - mix.loglike)/mix.loglike) < x_tol:
            return best_mix
        last_loglike = mix.loglike
    return best_mix  





