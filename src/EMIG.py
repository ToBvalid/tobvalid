# In[1]:
import math as math
import embase as embase
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import special
from scipy.special import erf
import scipy.stats as st
import seaborn as sns
import EMG as gm
import silverman as sv
sns.set_style("white")
import pheight as ph
from math import sqrt, log, exp, pi
from random import uniform
import bparser as bp
import pandas as pd


class InverseGammaDistribution(embase.Distribution):
    "Model univariate Inverse Gamma Distribution"
    def __init__(self, alpha, betta, shift):
        #alpha, betta and shiftself.
        self.alpha = alpha
        self.betta = betta
        self.shift = shift

    #probability density function
    def pdf(self, x):
        
        "Probability of a data point given the current parameters"
        if x <= self.shift:
            return 0
        if -self.betta / (x - self.shift) > 600:
            return 0

        y = ((x - self.shift)**(-self.alpha - 1)) * (self.betta**self.alpha) * math.exp(-self.betta / (x - self.shift)) / math.gamma(self.alpha) 
        return y

    def cdf(self, datum):
        if datum <= self.shift:
            return 0

        return special.gammaincc(self.alpha, self.betta/(datum - self.shift))
    #printing model values
    def __repr__(self):
        return 'InverseGamma({0:4.6}, {1:4.6}, {2:4.6})'.format(self.alpha, self.betta, self.shift)


#%%

class InverseGammaMixture(embase.Mixture):
    "Model mixture of N univariate Inverse Gamma Distributions and their EM estimation"

    def __init__(self, data, n, z, c = 0, step = 1, fisher = True, gm = None, modes = None, useMin = True):
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
        
        if gm != None:
            self.init3(gm)
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
        
    def init1(self):
        N = [sum(x) for x in z]
        mdB = [sum([z[i][j]*data[j] for j in range(len(self.data)) ])/N[i] for i in range(self.mode)]    
        varB = [sum([z[i][j]*(data[j] - mdB[i])**2 for j in range(len(data)) ])/N[i] for i in range(self.mode)]
        minB = [min([d for (d, zj) in zip(self.data, z[i]) if zj > self.c]) for i in range(self.mode)]
        self.initShift = minB
        
#        self.initShift.append(0.9*minB[0])
#        for i in range(self.mode - 1):
#            self.initShift.append((minB[i] + minB[i + 1])/2)
            
            
        self.initAlpha = [3.5]*self.mode
#        self.initShift =  [bm - sqrt(vb*(a - 2)) for (bm, vb, a) in zip(minB, varB, self.initAlpha)]
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
  
        
    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        # compute weights
        self.loglike = [0]*self.mode # = log(p = 1)
        sums = [sum([ m1*d1.pdf(datum) for (m1, d1) in zip(self.mix, self.dist)]) for datum in self.data]
        
        
        
        wps = [[m*d.pdf(datum) for (m, d) in zip(self.mix, self.dist)] for datum in self.data]
        for i in range(len(wps)):
            for j in range(len(wps[i])):
                if wps[i][j] != 0:
                    wps[i][j] = wps[i][j]/sums[i]
                
            
        self.weights = list(zip(* wps))
#        self.weights = []
#        weights = list(zip(* wps))
#        for i in range(self.mode):
#            w = []
#            for (d, wt) in zip(self.data, weights[i]):
#                if d < self.dist[i].shift:
#                    w.append(0)
#                else:
#                    w.append(wt)
#            self.weights.append(w)

        self.N = [sum(w) for w in self.weights]
        self.mix = [n/sum(self.N)  for n in self.N]

def Mstep(self):
        "Perform an M(aximization)-step"            
        result = []
        for i in range(0, self.mode):
            bmin = self.dist[i].shift
            conv = False
            self.CalcFisherMatrixForDist1(i)
            ss = self.shiftcalcForDist(i, self.epsilon)
            if (abs(ss) < [self.epsilon * x for x in np.absolute([self.dist[i].alpha, self.dist[i].betta, self.dist[i].shift])]).all():
                conv = True
            else:
                step = self.step
                alpha = self.dist[i].alpha
                betta = self.dist[i].betta
                shift = self.dist[i].shift
                fvalue = self.fvalue[i]

                ialpha = self.dist[i].alpha
                ibetta = self.dist[i].betta
                ishift = self.dist[i].shift
                ifvalue = self.fvalue[i]

                cycle = 0
                while  True:

                    self.dist[i].alpha = max(0.1, ialpha + step*ss[0, 0])
                    self.dist[i].betta = max(0.1, ibetta + step*ss[0, 1])
#                    
                    if self.useMin:
                        self.dist[i].shift = min(bmin, ishift + step*ss[0, 2])         
                    else:
                        self.dist[i].shift = ishift + step*ss[0, 2]       
#                    self.dist[i].alpha = max(0.1, self.dist[i].alpha + step*ss[0, 0])
#                    self.dist[i].betta = max(0.1, self.dist[i].betta + step*ss[0, 1])
#                    self.dist[i].shift = self.dist[i].shift + step*ss[0, 2]
                    #                    self.dist[i].shift = min(0.99*bmin, self.dist[i].shift + step*ss[0, 2])
#                    self.dist[i].shift = max(bmin, self.dist[i].shift + step*ss[0, 2])
#                    self.dist[i].shift = min(self.initShift[i], self.dist[i].shift + step*ss[0, 2])
#                    print(str(i) + " " + str(self.dist[i].shift) + " " + str(ss[2]))
#                    print(str(i) + " " + str(self.dist[i].shift) + " " + str(ss[0, 2]))
                    self.CalcFisherMatrixForDist1(i)

                    step = step*0.7

                    if conv and self.fvalue[i] >= fvalue:
                        self.dist[i].alpha = alpha
                        self.dist[i].betta = betta
                        self.dist[i].shift = shift
                        self.fvalue[i] = fvalue
                        break
                    
                    if self.fvalue[i] < fvalue:
                        fvalue = self.fvalue[i]
                        alpha = self.dist[i].alpha
                        betta = self.dist[i].betta
                        shift = self.dist[i].shift
                        cycle = cycle + 1
                        conv = True
  

                    
                    if  cycle > 100:
                        break
                    
                    
                    if step <= self.epsilon: 
                        if not conv:        
                            self.dist[i].alpha = ialpha
                            self.dist[i].betta = ibetta
                            self.dist[i].shift = ishift
                            self.fvalue[i] = ifvalue
                        break



#                    if self.fvalue[i] > fvalue:
#                        step = step*0.7
#                    else:
#                        conv = True
#                        break
#                    if step <= self.epsilon: 
#                        conv = True
#                        break
        
#                    self.dist[i].alpha = alpha
#                    self.dist[i].betta = betta
#                    self.dist[i].shift = shift
#                    self.fvalue[i] = fvalue

            result.append(conv)
            
        self.loglike = sum([m*f for (m,f) in zip(self.mix, self.fvalue)])
        return result
                
   
    def Mstep(self):
        "Perform an M(aximization)-step"            
        result = []
        for i in range(0, self.mode):
            bmin = self.dist[i].shift
            conv = False
            self.CalcFisherMatrixForDist1(i)
            ss = self.shiftcalcForDist(i, self.epsilon)
            if (abs(ss) < [self.epsilon * x for x in np.absolute([self.dist[i].alpha, self.dist[i].betta, self.dist[i].shift])]).all():
                conv = True
            else:
                step = self.step
                alpha = self.dist[i].alpha
                betta = self.dist[i].betta
                shift = self.dist[i].shift
                fvalue = self.fvalue[i]

                ialpha = self.dist[i].alpha
                ibetta = self.dist[i].betta
                ishift = self.dist[i].shift
                ifvalue = self.fvalue[i]

                cycle = 0
                while  True:

                    self.dist[i].alpha = max(0.1, ialpha + step*ss[0, 0])
                    self.dist[i].betta = max(0.1, ibetta + step*ss[0, 1])
#                    
                    if self.useMin:
                        self.dist[i].shift = min(bmin, ishift + step*ss[0, 2])         
                    else:
                        self.dist[i].shift = ishift + step*ss[0, 2]       
#                    self.dist[i].alpha = max(0.1, self.dist[i].alpha + step*ss[0, 0])
#                    self.dist[i].betta = max(0.1, self.dist[i].betta + step*ss[0, 1])
#                    self.dist[i].shift = self.dist[i].shift + step*ss[0, 2]
                    #                    self.dist[i].shift = min(0.99*bmin, self.dist[i].shift + step*ss[0, 2])
#                    self.dist[i].shift = max(bmin, self.dist[i].shift + step*ss[0, 2])
#                    self.dist[i].shift = min(self.initShift[i], self.dist[i].shift + step*ss[0, 2])
#                    print(str(i) + " " + str(self.dist[i].shift) + " " + str(ss[2]))
#                    print(str(i) + " " + str(self.dist[i].shift) + " " + str(ss[0, 2]))
                    self.CalcFisherMatrixForDist1(i)

                    step = step*0.7

                    if conv and self.fvalue[i] >= fvalue:
                        self.dist[i].alpha = alpha
                        self.dist[i].betta = betta
                        self.dist[i].shift = shift
                        self.fvalue[i] = fvalue
                        break
                    
                    if self.fvalue[i] < fvalue:
                        fvalue = self.fvalue[i]
                        alpha = self.dist[i].alpha
                        betta = self.dist[i].betta
                        shift = self.dist[i].shift
                        cycle = cycle + 1
                        conv = True
  

                    
                    if  cycle > 100:
                        break
                    
                    
                    if step <= self.epsilon: 
                        if not conv:        
                            self.dist[i].alpha = ialpha
                            self.dist[i].betta = ibetta
                            self.dist[i].shift = ishift
                            self.fvalue[i] = ifvalue
                        break



#                    if self.fvalue[i] > fvalue:
#                        step = step*0.7
#                    else:
#                        conv = True
#                        break
#                    if step <= self.epsilon: 
#                        conv = True
#                        break
        
#                    self.dist[i].alpha = alpha
#                    self.dist[i].betta = betta
#                    self.dist[i].shift = shift
#                    self.fvalue[i] = fvalue

            result.append(conv)
            
        self.loglike = sum([m*f for (m,f) in zip(self.mix, self.fvalue)])
        return result
                
   
      
    
    def CalcFisherMatrixForDist(self, i):
        d = self.dist[i]
        w = 1/self.sig**2
        n = self.N[i]
        zj = self.weights[i]
        
        minVal = 0.00000000001
        delta = [x - d.shift for x in self.data]
#        weightedDelta = sum([z/x for (z, x) in zip(zj, delta)])
        weightedLogDelta = sum([z*log(max(x, minVal)) for (z, x) in zip(zj, delta)])
        weightedInverseDelta = sum([z/x for (z, x) in zip(zj, delta)])
        logDelta = sum([log(max(x, minVal)) for x in delta])
        inverseDelta = sum([1/x for x in delta])
        inverseSquareDelta = sum([1/(x**2) for x in delta])
      
        
   
        self.fvalue[i] = - n*d.alpha*log(d.betta) + n*math.lgamma(d.alpha) + d.betta*weightedInverseDelta + (d.alpha + 1)*weightedLogDelta + w*(d.alpha - self.al0)**2/2 
        self.gradAlpha[i] = n*special.psi(d.alpha) - n*log(d.betta) + logDelta  + w*(d.alpha - self.al0)

        self.gradBetta[i] = -n*d.alpha/d.betta + inverseDelta
        
        self.gradB[i] = d.betta*inverseSquareDelta - (d.alpha + 1)*inverseDelta                                                                                                                         
    
        self.d2f[i] =  [[n*special.polygamma(1, d.alpha) + w, -n/d.betta, -n*d.alpha/d.betta ],
                 [-n/d.betta, n*d.alpha/d.betta**2, n*d.alpha*(d.alpha + 1)/d.betta**2],
              [-n*d.alpha/d.betta,  n*d.alpha*(d.alpha + 1)/d.betta**2,  n*d.alpha*(d.alpha + 1)*(d.alpha + 3)/d.betta**2]]
        
      
       
    def CalcFisherMatrixForDist1(self, i):
        d = self.dist[i]
        w = 1/self.sig**2
        n = self.N[i]
        zj = self.weights[i]
        
        delta = [(x - d.shift, z) for (x, z) in zip(self.data, zj)  if(x > d.shift)]
        
        weightedLogDelta = sum([z*log(x) for (x, z) in delta])
        weightedInverseDelta = sum([z/x for (x, z) in delta])
        logDelta = sum([log(x) for (x, z) in delta])
        inverseDelta = sum([z/x for (x, z)  in delta])
        inverseSquareDelta = sum([z/(x**2) for (x, z)  in delta])
        inverseCubeDelta = sum([z/(x**3) for (x, z) in delta])
        
   
        self.fvalue[i] = - n*d.alpha*log(d.betta) + n*math.lgamma(d.alpha) + d.betta*weightedInverseDelta + (d.alpha + 1)*weightedLogDelta + w*(d.alpha - self.al0)**2/2 
        self.gradAlpha[i] = n*special.psi(d.alpha) - n*log(d.betta) + logDelta  + w*(d.alpha - self.al0)

        self.gradBetta[i] = -n*d.alpha/d.betta + inverseDelta
        
        self.gradB[i] = d.betta*inverseSquareDelta - (d.alpha + 1)*inverseDelta                                                                                                                         
    
        if self.fisher == True:
            self.d2f[i] =  [[n*special.polygamma(1, d.alpha) + w, -n/d.betta, -n*d.alpha/d.betta ],
                     [-n/d.betta, n*d.alpha/d.betta**2, n*d.alpha*(d.alpha + 1)/d.betta**2],
                  [-n*d.alpha/d.betta,  n*d.alpha*(d.alpha + 1)/d.betta**2,  n*d.alpha*(d.alpha + 1)*(d.alpha + 3)/d.betta**2]]
        else:     
            self.d2f[i] =  [[n*special.polygamma(1, d.alpha) + w, -n/d.betta, -(d.alpha + 1)*inverseDelta],
                 [-n/d.betta, n*d.alpha/d.betta**2, -d.betta*inverseSquareDelta],
              [-inverseDelta,  inverseSquareDelta,  -2*d.betta*inverseCubeDelta + (d.alpha + 1)*inverseSquareDelta ]]
        
    def shiftcalcForDist(self, i, tol = 1.0e-4):
        u, s, vh = np.linalg.svd(self.d2f[i])
        l1 = len(np.linalg.svd(self.d2f[i]))
        ssinvd = [0, 0, 0] 
        mybul = s[0] * tol
        for j in range(0, l1):
            if s[j] > mybul:
                ssinvd[j] = 1/s[j]
                
        ssinv = np.mat(u) * np.mat(np.diag(ssinvd)) * np.mat(vh)
        shiftc = np.matmul(-ssinv, [self.gradAlpha[i], self.gradBetta[i], self.gradB[i]])
#        + tol*np.random.randn(3,3)
#        inv = np.linalg.inv(self.d2f[i] + tol*np.random.randn(3,3)) 
#        shiftc = np.matmul(-inv, [self.gradAlpha[i], self.gradBetta[i], self.gradB[i]])
        return shiftc
            
    def iterate(self, N=1, verbose=False):
        self.Estep()
        self.Mstep()
        
    def __repr__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])




def igmm(data, mode, z, z_tol = 0.1, max_iter = 100, x_tol = 0.01, step = 1, fisher = True, gm=None, modes = None, useMin = True):
    last_loglike = float('-inf')
    mix = InverseGammaMixture(data, mode, z, z_tol, step = step, fisher = fisher, gm = gm, modes = modes, useMin=useMin)
    for i in range(max_iter):
#        try:
        mix.iterate()
        if abs((last_loglike - mix.loglike)/mix.loglike) < x_tol:
            return mix
        last_loglike = mix.loglike
#        except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
#            print("ERROR " + str(i))
#            pass
    return mix
    


def analysis(file, max_iter=100, z_tol = 0.1):
    (s, data) = bp.bfactor_parse(file)
    p_data = ph.peak_height(data, s)
    nmodes = sv.boot_silverman(p_data, length = 2000)[0]
    gmmres =  gm.gmm(p_data, nmodes)
    z = gmmres.z
    z.reverse()
    return (igmm(data, nmodes, z, z_tol=z_tol, max_iter=max_iter), data)
    
    


(s, data) = bp.bfactor_parse("../data/bfactor/1RXM_out.txt")
p_data = ph.peak_height(data, s)
nmodes = sv.boot_silverman(p_data, length = 2000)[0]
print(nmodes)

gmmres =  gm.gmm(p_data, nmodes)
z = gmmres.z


plt.subplot(3, 1, 1)
for i in range(nmodes):
    a = np.transpose(np.array([p_data, z[i]]))
    sa = a[a[:,0].argsort()]
    plt.plot(sa[:, 0], sa[:, 1], label=str(i + 1) + 'z')

x1 = np.linspace(start=min(p_data), stop=max(p_data), num=1000)
sns.distplot(p_data, bins=20, kde=False, norm_hist=True)
dists = [[gmmres.mix[i]*gmmres.dist[i].pdf(e) for e in x1]  for i in range(nmodes)]
#all_dist = [gmmres.pdf(e) for e in x2]
for i in range(nmodes):
    plt.plot(x1, dists[i], label=str(i + 1))
#plt.plot(x1, all_dist, label='Inverse Gamma mixture')
plt.legend()

#z.reverse()
#best_mix= igmm(data, nmodes, z, z_tol=0.9, max_iter=40, x_tol=0.0001, step = 1, fisher = True)
best_gm = gm.gmm(data, 2, max_iter = 100)
#(best_mix, data) = analysis("../data/bfactor/1RXM_out.txt", z_tol=0.1, max_iter=20)
print(best_gm)    




plt.subplot(3, 1, 2)
x2 = np.linspace(start=min(data), stop=max(data), num=1000)
sns.distplot(data, bins=20, kde=False, norm_hist=True)
dists = [[best_gm.mix[i]*best_gm.dist[i].pdf(e) for e in x2]  for i in range(nmodes)]
all_dist = [best_gm.pdf(e) for e in x2]
for i in range(nmodes):
    plt.plot(x2, dists[i], label=str(i + 1))
plt.plot(x2, all_dist, label='Gaussian Gamma mixture')
plt.legend()



z = best_gm.z
z = gmmres.z
z.reverse()

best_mix= igmm(data,  nmodes,  z, z_tol=0.1, max_iter=20, x_tol=0.00000000001, step = 1, fisher = False, modes = [15, 35], useMin=False)
print(best_mix)


plt.subplot(3, 1, 3)
x3 = np.linspace(start=min(data), stop=max(data), num=1000)
sns.distplot(data, bins=20, kde=False, norm_hist=True)
idists = [[best_mix.mix[i]*best_mix.dist[i].pdf(e) for e in x3]  for i in range(nmodes)]
iall_dist = [best_mix.pdf(e) for e in x3]
for i in range(nmodes):
    plt.plot(x3, idists[i], label=str(i + 1))
plt.plot(x3, iall_dist, label='Inverse Gamma mixture')
plt.legend()


print(best_mix.initBetta)
print([d.mu - d.sigma for d in best_gm.dist])
print([d.mu for d in best_gm.dist]) 
print([d.sigma for d in best_gm.dist])  
        
#print(best_mix.initAlpha)
#
#
#
##%%
#def Peak_height(B, s):
#    smax = 1/float(s)
#    B = (B-np.amin(B)) + 0.01
#    rho_zero = (8*np.pi/B)*(-smax*np.exp(-B*smax**2/4)+np.sqrt(np.pi/B)*erf(np.sqrt(B)*smax/2))
#    rho_zero = rho_zero /max(rho_zero)
#    return(rho_zero)
##%%
#data = Peak_height(igData, 3)
#sns.distplot(data, bins=20, kde=False)
##%%
##%%
## Find best Mixture Gaussian model
#n_iterations = 10
#n_random_restarts = 1
#best_mix1 = None
#best_loglike1 = float('-inf')
#print('Computing best model with random restarts...\n')
#result = None
#for _ in range(n_random_restarts):
#    mix = InverseGammaMixture(igData, 2, [0.2, 0.8])
#    result = mix.iterate(n_iterations)
#
#    if all(result):
#        best_mix1 = mix
#        break
#    best_mix1 = mix
##     try:
#  
## #             print(mix.loglike)
##     except ZeroDivisionError: # Catch division errors from bad starts, and just throw them out...
##         print("ZeroDivisionError")
##     except ValueError: # Catch division errors from bad starts, and just throw them out...
##         print("ZeroDivisionError")
##     except RuntimeWarning: # Catch division errors from bad starts, and just throw them out...
##         print("RuntimeWarning")
#print(result)  
#print(best_mix1)
#print('\n\nDone. 🙂') 