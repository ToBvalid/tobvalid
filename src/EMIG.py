# In[1]:
import math as math

#For plotting
import matplotlib.pyplot as plt
#for matrix math
import numpy as np
#for normalization + probability density function computation
from scipy import stats
#for plotting
from scipy import special
from scipy.special import erf
import scipy.stats as st
import seaborn as sns
from scipy import special
sns.set_style("white")


from math import sqrt, log, exp, pi
from random import uniform
#for data preprocessing
import pandas as pd

class Distribution:
    def pdf(self, x):
        pass
    def cdf(self, x):
        pass
    def __repr__(self):
        pass

class Mixture:
    def __init__(self, data, n):
        self.mode = n
        self.data = data
        self.loglike = 0.
        self.mix = [1/self.mode]*self.mode

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
    
    def cdf(self, x):
        return sum([w*d.cdf(x) for (w, d) in zip (self.mix, self.dist)])

    def __repr__(self):
        pass

    def __str__(self):
        pass


# In[4]:
class InverseGammaDistribution(Distribution):
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

class InverseGammaMixture(Mixture):
    "Model mixture of N univariate Inverse Gamma Distributions and their EM estimation"

    def __init__(self, data, n, shift):
        self.al0 = 3.5
        self.sig = 0.1
        self.epsilon = 1.0e-16
        super().__init__(data, n)
        self.initShift = []
        

        self.initShift.append(0.9*min(data))
        for i in range(self.mode - 1):
            self.initShift.append((shift[i] + shift[i + 1])/2)
            
        self.initAlpha = [3.5]*self.mode
        self.initBetta = [(bm - b)*(a + 1) for (bm, b, a) in zip(shift, self.initShift, self.initAlpha)]
        self.dist = [InverseGammaDistribution(a, b, shift) for (a, b, shift) in zip(self.initAlpha, self.initBetta, self.initShift)]
    
                
        self.fvalue = [0]*self.mode
        
        self.gradAlpha = [0]*self.mode
        
        self.gradBetta = [0]*self.mode
        
        self.gradB = [0]*self.mode                                                                                                                       
        
        self.d2f =  [0]*self.mode 
        
    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        # compute weights
        self.loglike = [0]*self.mode # = log(p = 1)
        sums = [sum([ m1*d1.pdf(datum) for (m1, d1) in zip(self.mix, self.dist)]) for datum in self.data]
        wps = [[m*d.pdf(datum)/s
                for (m, d) in zip(self.mix, self.dist)] for (datum, s) in zip(self.data, sums)]
        self.weights = list(zip(* wps))

        self.N = [sum(w) for w in self.weights]
        self.mix = [n/sum(self.N)  for n in self.N]

    def Mstep(self):
        "Perform an M(aximization)-step"        
        
        result = []
        for i in range(0, self.mode):
            bmin = self.initShift[i]
            conv = False
            self.CalcFisherMatrixForDist(i)
            ss = self.shiftcalcForDist(i, self.epsilon)
            if (abs(ss) < [self.epsilon * x for x in np.absolute([self.dist[i].alpha, self.dist[i].betta, self.dist[i].shift])]).all():
                conv = True
            else:
                step = 1
                alpha = self.dist[i].alpha
                betta = self.dist[i].betta
                shift = self.dist[i].shift
                fvalue = self.fvalue[i]

                while  not conv:
                    self.dist[i].alpha = max(0.1, self.dist[i].alpha + step*ss[0, 0])
                    self.dist[i].betta = max(0.1, self.dist[i].betta + step*ss[0, 1])
                    self.dist[i].shift = min(0.99*bmin, self.dist[i].shift + step*ss[0, 2])
                         
                    self.CalcFisherMatrixForDist(i)

                    if self.fvalue[i] > fvalue:
                        step = step*0.7
                    else:
                        conv = True
                        break
                    if step <= self.epsilon: 
                        conv = True
                        break
        
                    self.dist[i].alpha = alpha
                    self.dist[i].betta = betta
                    self.dist[i].shift = shift
                    self.fvalue[i] = fvalue

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
        
        minVal = 0.00000000001
        delta = list(filter(lambda a: a > 0, [x - d.shift for x in self.data]))
        
#        weightedDelta = sum([z/x for (z, x) in zip(zj, delta)])
        weightedLogDelta = sum([z*log(x) for (z, x) in zip(zj, delta)])
        weightedInverseDelta = sum([z/x for (z, x) in zip(zj, delta)])
        logDelta = sum([log(x) for x in delta])
        inverseDelta = sum([1/x for x in delta])
        inverseSquareDelta = sum([1/(x**2) for x in delta])
        
        
   
        self.fvalue[i] = - n*d.alpha*log(d.betta) + n*math.lgamma(d.alpha) + d.betta*weightedInverseDelta + (d.alpha + 1)*weightedLogDelta + w*(d.alpha - self.al0)**2/2 
        self.gradAlpha[i] = n*special.psi(d.alpha) - n*log(d.betta) + logDelta  + w*(d.alpha - self.al0)

        self.gradBetta[i] = -n*d.alpha/d.betta + inverseDelta
        
        self.gradB[i] = d.betta*inverseSquareDelta - (d.alpha + 1)*inverseDelta                                                                                                                         
    
        self.d2f[i] =  [[n*special.polygamma(1, d.alpha) + w, -n/d.betta, -n*d.alpha/d.betta ],
                 [-n/d.betta, n*d.alpha/d.betta**2, n*d.alpha*(d.alpha + 1)/d.betta**2],
              [-n*d.alpha/d.betta,  n*d.alpha*(d.alpha + 1)/d.betta**2,  n*d.alpha*(d.alpha + 1)*(d.alpha + 3)/d.betta**2]]
        
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
        return shiftc
            
    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"
        loglike = 0
        result = []
        for iter  in range(0, N) :
            print(self.loglike)
            self.Estep()
            result = self.Mstep()
#             if loglike >= self.loglike:
#                 print(iter)
#                 break
            loglike = self.loglike
            
        return result
        
    def __repr__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])


#%%
#read our dataset
igDt = pd.read_csv("1B59_out.txt")
#show first 5 examples (in BTC)
igDt.head(n=5)
# In[3]:
igData = igDt.x
sns.distplot(igData, bins=20, kde=False)

#%%
# Find best Mixture Gaussian model
n_iterations = 10
n_random_restarts = 1
best_mix = None
best_loglike = float('-inf')
print('Computing best model with random restarts...\n')
result = None
for _ in range(n_random_restarts):
    mix = InverseGammaMixture(igData, 2, [30, 110])
    result = mix.iterate(n_iterations)

    if all(result):
        best_mix = mix
        break
    best_mix = mix
#     try:
print(result)  
print(best_mix)
print('\n\nDone. 🙂')

#%%
#mixture
#Return evenly spaced numbers over a specified interval.
x = np.linspace(start=0.00534*-0, stop=200, num=1000)
sns.distplot(igData, bins=20, kde=False, norm_hist=True)
g_both = [best_mix.pdf(e) for e in x]
plt.plot(x, g_both, label='Inverse Gamma mixture')
plt.legend()


#%%
histo, bin_edges = np.histogram(igData, bins='auto', normed=False)
number_of_bins = len(bin_edges) - 1
observed_values = histo
cdf = [best_mix.cdf(d)  for d in bin_edges]
expected_values = len(igData) * np.diff(cdf)
c , p = st.chisquare(observed_values, expected_values)
print(c)
print(p)

#%%
def Peak_height(B, s):
    smax = 1/float(s)
    B = (B-np.amin(B)) + 0.01
    rho_zero = (8*np.pi/B)*(-smax*np.exp(-B*smax**2/4)+np.sqrt(np.pi/B)*erf(np.sqrt(B)*smax/2))
    rho_zero = rho_zero /max(rho_zero)
    return(rho_zero)
#%%
data = Peak_height(igData, 3)
sns.distplot(data, bins=20, kde=False)
#%%
#%%
# Find best Mixture Gaussian model
n_iterations = 10
n_random_restarts = 1
best_mix1 = None
best_loglike1 = float('-inf')
print('Computing best model with random restarts...\n')
result = None
for _ in range(n_random_restarts):
    mix = InverseGammaMixture(igData, 2, [0.2, 0.8])
    result = mix.iterate(n_iterations)

    if all(result):
        best_mix1 = mix
        break
    best_mix1 = mix
#     try:
  
# #             print(mix.loglike)
#     except ZeroDivisionError: # Catch division errors from bad starts, and just throw them out...
#         print("ZeroDivisionError")
#     except ValueError: # Catch division errors from bad starts, and just throw them out...
#         print("ZeroDivisionError")
#     except RuntimeWarning: # Catch division errors from bad starts, and just throw them out...
#         print("RuntimeWarning")
print(result)  
print(best_mix1)
print('\n\nDone. 🙂') 