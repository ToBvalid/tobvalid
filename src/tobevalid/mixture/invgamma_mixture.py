# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:26:01 2019

@author: KavehB
"""



import numpy as np
from scipy import special
import scipy.stats as st

from .base import BaseMixture


class InverseGammaMixture(BaseMixture):
     def __init__(self, n_modes = 1, tol = 1e-3, max_iter = 100):
        BaseMixture.__init__(self, n_modes, tol, max_iter) 
     
     def _check_initial_custom_parameters(self, **kwargs):
        return
    
     def _check_parameters(self, X, **kwargs):
        return
     
     
     def _init_parameters(self, **kwargs):
        self.c = 0.1
        self.al0 = 3.5
        self.sig = 0.1
        self.epsilon = 1.0e-16
        self.step = 1
         
        self.Z = kwargs["z"]
        N = self.Z.sum(axis = 0)
        mdB = np.matmul(self.data, self.Z)/N
        minB =  [np.min(self.data[self.Z[:, i] > self.c]) for i in range(self.n_modes)] 
        self.shift = np.array(minB)
        self.alpha = np.full(self.n_modes, 3.5)
        self.betta = (mdB - self.shift)*(self.alpha - 1)
        

        
     def _m_step(self):
        self.CalcFisherMatrix()
        ss = self.shiftcalc(self.epsilon)
        step = self.step
        ialpha = self.alpha.copy() 
        ibetta = self.betta.copy() 
        ishift = self.shift.copy()
        ifvalue = self._loglike
        conv = False
        while not conv:
            self.alpha = np.maximum(ialpha + step*ss[0], np.array([0.1]*self.n_modes))
            self.betta = np.maximum(ibetta + step*ss[1], np.array([0.1]*self.n_modes))
            self.shift = ishift + step*ss[2]       

            self.CalcFisherMatrix()

            if self._loglike > ifvalue:
                step = step*0.7
            else:
                conv = True
                break
            if step <= self.epsilon: 
                if not conv:
                    for i in range(self.n_modes):
                        self.alpha[i] = ialpha[i]
                        self.betta[i] = ibetta[i]
                        self.shift[i]= ishift[i]
                    self._loglike = ifvalue
                break 
        return conv
      
     def CalcFisherMatrix(self):
        w = np.array([1/self.sig**2] + [0]*(self.n_modes - 1)) 

        self._loglike = 0
        self.gradA = 0
        self.gradBt = 0
        self.gradS = 0
        self.d2l = np.zeros((3*self.n_modes, 3*self.n_modes))
        self.fvalue = [0]*self.n_modes
        
        self.gradAlpha = [0]*self.n_modes
        
        self.gradBetta =  [0]*self.n_modes
        
        self.gradB =   [0]*self.n_modes  
        self.d2f =  [0]*self.n_modes 
        weightedLogDelta = np.zeros((self.n_modes))
        weightedInverseDelta = np.zeros((self.n_modes))
        inverseSquareDelta = np.zeros((self.n_modes))
        
        for i in range(self.n_modes):   

            n = self.N[i]

            zj = self.Z.T[i]    
            idx = self.data > self.shift[i]
            delta = self.data[idx] - self.shift[i]
            
            zj = zj[idx] 
            
            weightedLogDelta[i] = np.dot(zj, np.log(delta))
            weightedInverseDelta[i] = np.sum(zj/delta)
            inverseSquareDelta[i] = np.sum(zj/delta**2)
            
            alpha = self.alpha[i]
            betta = self.betta[i]
            self.d2f[i] =  np.array([[n*special.polygamma(1, alpha) + w[0], -n/betta, -n*alpha/betta ],
                         [-n/betta, n*alpha/np.power(betta, 2), n*alpha*(alpha + 1)/np.power(betta, 2)],
                      [-n*alpha/betta,  n*alpha*(alpha + 1)/np.power(betta, 2),  n*alpha*(alpha + 1)*(alpha + 3)/np.power(betta, 2)]])

        
            self.gradAlpha = self.N*special.psi(self.alpha) - self.N*np.log(self.betta) + weightedLogDelta  + w*(self.alpha - self.al0)  
            self.gradBetta = -self.N*self.alpha/self.betta + weightedInverseDelta        
            self.gradB = self.betta*inverseSquareDelta - (self.alpha + 1)*weightedInverseDelta         
                     
            self._loglike = np.dot(-self.N*self.alpha*np.log(self.betta) + self.N*special.gammaln(self.alpha) + self.betta*weightedInverseDelta + (self.alpha + 1)*weightedLogDelta  + w*(self.alpha - self.al0)**2/2, self.mix)

        for i in range(self.n_modes):
            for j in range(3):
                for k in range(3):
                    self.d2l[3*i + j, k + 3*i] =  self.mix[i]*self.d2f[i][j][k]
                  
     def shiftcalc(self, tol = 1.0e-5):

        inv = np.linalg.inv(self.d2l + tol*np.random.randn(3*self.n_modes, 3*self.n_modes))
        grad = np.transpose(np.array([self.gradAlpha, self.gradBetta, self.gradB])).flatten()
        shiftc = np.matmul(-inv, grad)
        return shiftc.reshape((self.n_modes, 3)).T
        return True
        
     def _pdf(self, X):
         dist = st.invgamma(self.alpha, self.shift, self.betta)
         return dist.pdf(X)