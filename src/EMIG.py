import math as math
import embase as embase
import numpy as np
from scipy import special
import scipy.stats as st
from math import log

class InverseGammaDistribution(embase.Distribution):
    "Model univariate Inverse Gamma Distribution"
    def __init__(self, alpha, betta, shift):
        #alpha, betta and shiftself.
        self.alpha = alpha
        self.betta = betta
        self.shift = shift
        self.dist = getattr(st, 'invgamma')

    #probability density function
    def pdf(self, x):
        
        "Probability of a data point given the current parameters"
        return self.dist.pdf(x, self.alpha, loc=self.shift, scale=self.betta)

    def cdf(self, datum):
        if datum <= self.shift:
            return 0

        return special.gammaincc(self.alpha, self.betta/(datum - self.shift))
    
    def size(self):
        return 3
    def setParams(self, params):
        self.alpha = max(0.1, params[0])
        self.betta = max(0.1,params[1])
        self.shift = params[2]
    def getParams(self):
        return [self.alpha, self.betta, self.shift]        
    #printing model values
    def __repr__(self):
        return 'InverseGamma({0:4.6}, {1:4.6}, {2:4.6})'.format(self.alpha, self.betta, self.shift)



class InverseGammaMixture(embase.Mixture):
    "Model mixture of N univariate Inverse Gamma Distributions and their EM estimation"

    def __init__(self, data, n, z, c = 0, step = 1, fisher = True, gm = None, modes = None, useMin = True, initial=None, mix=None):
        self.al0 = 3.5
        self.sig = 0.1
        self.epsilon = 1.0e-16
        super().__init__(data, n)
        self.initShift = []
        self.step = step
        self.fisher = fisher
        self.c = c
        self.z = z
        self.useMin = useMin
        
#        if mix.all() != None:
#            self.mix = mix
        if gm != None:
            self.init3(gm)
        elif not (initial is None):
            self.init5(initial)
        elif   modes != None:
            self.init4(modes)
        else:
            self.init1()
        
        self.dist = [InverseGammaDistribution(a, b, shift) for (a, b, shift) in zip(self.initAlpha, self.initBetta, self.initShift)]
    
                
        self.fvalue = [0]*self.mode
        
        self.gradAlpha = [0]*self.mode
        
        self.gradBetta =  [0]*self.mode
        
        self.gradB =   [0]*self.mode                                                                                                                    
        
        self.d2f =  [0]*self.mode 
        
        
        self.loglike = 0
        self.gradA = 0
        self.gradBt = 0
        self.gradS = 0
        self.d2l = np.zeros((3*self.mode, 3*self.mode))
        
    def init1(self):
        N = [sum(x) for x in self.z]
        mdB = [sum([self.z[i][j]*self.data[j] for j in range(len(self.data)) ])/N[i] for i in range(self.mode)]    
#        varB = [sum([self.z[i][j]*(self.data[j] - mdB[i])**2 for j in range(len(self.data)) ])/N[i] for i in range(self.mode)]
        minB = [min([d for (d, zj) in zip(self.data, self.z[i]) if zj > self.c]) for i in range(self.mode)]
        self.initShift = minB
        self.initAlpha = [3.5]*self.mode
        self.initBetta = [(bm - b)*(a - 1) for (bm, b, a) in zip(mdB, self.initShift, self.initAlpha)]
    def init2(self):
        dat = [[d for (d, zj) in  zip(self.data, self.z[i]) if zj > self.c]  for i in range(self.mode)]
        mean = [np.mean(d) for d in dat]
        var = [np.var(d) for d in dat]
        minB = [min([d for (d, zj) in zip(self.data, self.z[i]) if zj > self.c]) for i in range(self.mode)]

        self.initShift = minB
        
        self.initAlpha = [ m**2/v + 2 for (m, v) in zip(mean, var)]
        self.initBetta = [ m*(m**2/v + 1) for (m, v) in zip(mean, var)]
    
    def init3(self, gm):
        self.initShift = [d.mu - 3*d.sigma for d in gm.dist]
        
        self.initAlpha = [d.mu**2/d.sigma + 2 for d in gm.dist] 
        self.initBetta = [(d.mu**2/d.sigma + 1)*d.sigma for d in gm.dist] 
        
    def init4(self, modes):
        self.initShift = []
        self.initShift.append(0.9*modes[0])
        for i in range(len(modes) - 1):
            self.initShift.append((modes[i] + modes[i + 1])/2)
         
        self.initAlpha = [3.5]*self.mode
        self.initBetta = [(bm - b)*(a + 1) for (bm, b, a) in zip(modes, self.initShift, self.initAlpha)]
        
    def init5(self, initial):
        self.initShift = initial[1]
        self.initAlpha = initial[0]
        self.initBetta = initial[2]
  
        
    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        # compute weights
#        sums = [sum([ m1*d1.pdf(datum) for (m1, d1) in zip(self.mix, self.dist)]) for datum in self.data]
#        
#        
#
#        wps = [[m*d.pdf(datum) for (m, d) in zip(self.mix, self.dist)] for datum in self.data]
#        for i in range(len(wps)):
#            for j in range(len(wps[i])):
#                if wps[i][j] != 0:
#                    wps[i][j] = wps[i][j]/sums[i]
#                
#                    
#        self.weights = list(zip(* wps))
#        self.N = [sum(w) for w in self.weights]
#        self.mix = [n/sum(self.N)  for n in self.N]
        
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
            result = []
            

            self.CalcFisherMatrix()
            ss = self.shiftcalc(self.epsilon)
            step = self.step
            ialpha = [d.alpha for d in self.dist] 
            ibetta = [d.betta for d in self.dist] 
            ishift = [d.shift for d in self.dist]
            ifvalue = self.loglike
            conv = False
            while not conv:
                for i in range(self.mode):
             
                    self.dist[i].alpha = max(0.1, ialpha[i] + step*ss[0 + 3*i])
                    self.dist[i].betta = max(0.1, ibetta[i] + step*ss[1 + 3*i])
                    self.dist[i].shift = ishift[i] + step*ss[2 + 3*i]       
                
               
                self.CalcFisherMatrix()
                if self.loglike > ifvalue:
                    step = step*0.7
                else:
                    conv = True
               
                if step <= self.epsilon: 
                    if not conv:
                        for i in range(self.mode):
                            self.dist[i].alpha = ialpha[i]
                            self.dist[i].betta = ibetta[i]
                            self.dist[i].shift = ishift[i]
                        self.loglike = ifvalue
                    break
            result.append(conv)
            return result
      
    def CalcFisherMatrix(self):
        w = 1/self.sig**2 
        self.loglike = 0
        self.gradA = 0
        self.gradBt = 0
        self.gradS = 0
        self.d2l = np.zeros((3*self.mode, 3*self.mode))

        for i in range(self.mode):   
            d = self.dist[i]    
            n = self.N[i]
            zj = self.weights[i]    
            delta = [(x - d.shift, z) for (x, z) in zip(self.data, zj)  if(x > d.shift)]
            
            weightedLogDelta = sum([z*log(x) for (x, z) in delta])
            weightedInverseDelta = sum([z/x for (x, z) in delta])
            logDelta = sum([log(x) for (x, z) in delta])
            inverseDelta = sum([z/x for (x, z)  in delta])
            inverseSquareDelta = sum([z/(x**2) for (x, z)  in delta])
            inverseCubeDelta = sum([z/(x**3) for (x, z) in delta])
            
           
            self.fvalue[i] = - n*d.alpha*log(d.betta) + n*math.lgamma(d.alpha) + d.betta*weightedInverseDelta + (d.alpha + 1)*weightedLogDelta  + w*(d.alpha - self.al0)**2/2 
            self.gradAlpha[i] = n*special.psi(d.alpha) - n*log(d.betta) + weightedLogDelta  + w*(d.alpha - self.al0)
            self.gradBetta[i] = -n*d.alpha/d.betta + inverseDelta
            
            self.gradB[i] = d.betta*inverseSquareDelta - (d.alpha + 1)*inverseDelta                                                                                                                         
        
            if self.fisher == True:
                self.d2f[i] =  np.array([[n*special.polygamma(1, d.alpha) + w, -n/d.betta, -n*d.alpha/d.betta ],
                         [-n/d.betta, n*d.alpha/d.betta**2, n*d.alpha*(d.alpha + 1)/d.betta**2],
                      [-n*d.alpha/d.betta,  n*d.alpha*(d.alpha + 1)/d.betta**2,  n*d.alpha*(d.alpha + 1)*(d.alpha + 3)/d.betta**2]])
            else:     
                self.d2f[i] = np.array( [[n*special.polygamma(1, d.alpha) + w, -n/d.betta, -(d.alpha + 1)*inverseDelta],
                     [-n/d.betta, n*d.alpha/d.betta**2, -d.betta*inverseSquareDelta],
                  [-inverseDelta,  inverseSquareDelta,  -2*d.betta*inverseCubeDelta + (d.alpha + 1)*inverseSquareDelta ]])
                
                 
    
        self.loglike = sum([self.mix[i]*self.fvalue[i] for i in range(self.mode)])
        self.gradA = sum([self.mix[i]*self.gradAlpha[i] for i in range(self.mode)])
        self.gradBt = sum([self.mix[i]*self.gradBetta[i] for i in range(self.mode)])
        self.gradS = sum([self.mix[i]*self.gradB[i] for i in range(self.mode)])
        for i in range(self.mode):
            for j in range(3):
                for k in range(3):
                    self.d2l[3*i + j, k + 3*i] =  self.mix[i]*self.d2f[i][j][k]
        
        
    def shiftcalc(self, tol = 1.0e-4):
#        u, s, vh = np.linalg.svd(self.d2l)
#        l1 = len(np.linalg.svd(self.d2l))
#        ssinvd = [0, 0, 0] 
#        mybul = s[0] * tol
#        for j in range(0, l1):
#            if s[j] > mybul:
#                ssinvd[j] = 1/s[j]
#                
#        ssinv = np.mat(u) * np.mat(np.diag(ssinvd)) * np.mat(vh)
#        shiftc = np.matmul(-ssinv, [self.gradA, self.gradBt, self.gradS])
#        + tol*np.random.randn(3,3)

        
        inv = np.linalg.inv(self.d2l + tol*np.random.randn(3*self.mode, 3*self.mode)) 
        grad = np.transpose(np.array([self.gradAlpha, self.gradBetta, self.gradB])).flatten()
        shiftc = np.matmul(-inv, grad)
        return shiftc
            
    def iterate(self, N=1, verbose=False):
        self.Estep()
        self.Mstep()
        
    def __repr__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])




def igmm(data, mode, z, z_tol = 0.1, max_iter = 100, x_tol = 0.0000001, step = 1, fisher = True, gm=None, modes = None, useMin = True, initial=None, mix=None):
    last_loglike = float('inf')
    mix = InverseGammaMixture(data, mode, z, z_tol, step = step, fisher = fisher, gm = gm, modes = modes, useMin=useMin, initial=initial, mix = mix)
    best_mix = mix
    for i in range(max_iter):
        mix.iterate()
        if(last_loglike < mix.loglike):
            return best_mix
        if abs((last_loglike - mix.loglike)/mix.loglike) < x_tol:
            return mix
        last_loglike = mix.loglike
    return mix
    

#
#def analysis(file, max_iter=100, z_tol = 0.1):
#    (s, data) = bp.bfactor_parse(file)
#    p_data = ph.peak_height(data, s)
#    nmodes = sv.boot_silverman(p_data, length = 2000)[0]
#    gmmres =  gm.gmm(p_data, nmodes)
#    z = gmmres.z
#    z.reverse()
#    return (igmm(data, nmodes, z, z_tol=z_tol, max_iter=max_iter), data)
    
    
