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

    def __init__(self, data, n, z, c = 0, step = 1):
        self.al0 = 3.5
        self.sig = 0.1
        self.epsilon = 1.0e-16
        embase.Mixture.__init__(self, data, n)
        self.initShift = []
        self.step = step
        self.c = c
        self.z = z

        self.init()
        self.dist = [InverseGammaDistribution(a, b, shift) for (a, b, shift) in zip(self.alpha, self.betta, self.shift)]

        self.loglike = 0

        
    def init(self):
        N = self.z.sum(axis = 1)  
        mdB = np.matmul(self.z, self.data.T)/N
        minB =  [np.min(self.data[self.z[i] > self.c]) for i in range(self.mode)]

        self.shift = np.array(minB)
        self.alpha = np.full(self.mode, 3.5)
        self.betta = (mdB - self.shift)*(self.alpha - 1)
   
        
    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"

        pdfs = [np.vectorize(d.pdf) for d in self.dist]
        values = np.array([f(self.data) for f in pdfs]) 
        sums = np.matmul(self.mix, values)
        wps = (values.T*self.mix).T
   
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
            ialpha = self.alpha.copy() 
            ibetta = self.betta.copy() 
            ishift = self.shift.copy()
            ifvalue = self.loglike
            conv = False
            while not conv:
                self.alpha = ialpha + step*ss[0]
                self.betta = ibetta + step*ss[1]
                self.shift = ishift + step*ss[2]       

              
                self.CalcFisherMatrix()
                if self.loglike > ifvalue:
                    step = step*0.7
                else:
                    conv = True
               
                if step <= self.epsilon: 
                    if not conv:
                        for i in range(self.mode):
                            self.alpha[i] = ialpha[i]
                            self.betta[i] = ibetta[i]
                            self.shift[i]= ishift[i]
                        self.loglike = ifvalue
                    break
            for i in range(self.mode):
                self.dist[i].alpha = self.alpha[i]
                self.dist[i].betta = self.betta[i]
                self.dist[i].shift= self.shift[i]
                
            return conv
      
    def CalcFisherMatrix(self):
        w = np.array([1/self.sig**2] + [0]*(self.mode - 1)) 
        self.loglike = 0
        self.gradA = 0
        self.gradBt = 0
        self.gradS = 0
        self.d2l = np.zeros((3*self.mode, 3*self.mode))
        self.fvalue = [0]*self.mode
        
        self.gradAlpha = [0]*self.mode
        
        self.gradBetta =  [0]*self.mode
        
        self.gradB =   [0]*self.mode  
        self.d2f =  [0]*self.mode 
        
        weightedLogDelta = np.zeros((self.mode))
        weightedInverseDelta = np.zeros((self.mode))
        inverseSquareDelta = np.zeros((self.mode))
        for i in range(self.mode):   

            n = self.N[i]
            zj = self.weights[i]    
            idx = self.data > self.shift[i]
            delta = self.data[idx] - self.shift[i]
            zj = zj[idx] 
            weightedLogDelta[i] = np.dot(zj, np.log(delta))
            weightedInverseDelta[i] = np.sum(zj/delta)
            inverseSquareDelta[i] = np.sum(zj/delta**2)
            
            alpha = self.alpha[i]
            betta = self.betta[i]
            self.d2f[i] =  np.array([[n*special.polygamma(1, alpha) + w[0], -n/betta, -n*alpha/betta ],
                         [-n/betta, n*alpha/betta**2, n*alpha*(alpha + 1)/betta**2],
                      [-n*alpha/betta,  n*alpha*(alpha + 1)/betta**2,  n*alpha*(alpha + 1)*(alpha + 3)/betta**2]])
        
        self.gradAlpha = self.N*special.psi(self.alpha) - self.N*np.log(self.betta) + weightedLogDelta  + w*(self.alpha - self.al0)  
        self.gradBetta = -self.N*self.alpha/self.betta + weightedInverseDelta        
        self.gradB = self.betta*inverseSquareDelta - (self.alpha + 1)*weightedInverseDelta         
                 
    
        self.loglike = np.dot(-self.N*self.alpha*np.log(self.betta) + self.N*special.gammaln(self.alpha) + self.betta*weightedInverseDelta + (self.alpha + 1)*weightedLogDelta  + w*(self.alpha - self.al0)**2/2, self.mix)
        for i in range(self.mode):
            for j in range(3):
                for k in range(3):
                    self.d2l[3*i + j, k + 3*i] =  self.mix[i]*self.d2f[i][j][k]
    def shiftcalc(self, tol = 1.0e-4):
        inv = np.linalg.inv(self.d2l + tol*np.random.randn(3*self.mode, 3*self.mode)) 
        grad = np.transpose(np.array([self.gradAlpha, self.gradBetta, self.gradB])).flatten()
        shiftc = np.matmul(-inv, grad)
        return shiftc.reshape((self.mode, 3)).T
            
    def iterate(self, N=1, verbose=False):
        self.Estep()
        return self.Mstep()
        
    def __repr__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])
    
    def __str__(self):
        return ''.join(['InverseGammaMixture({0}, mix={1:.03}) '.format(self.dist[i], self.mix[i]) for i in range(self.mode)])

    def logLike(self, i, alpha, betta, shift):
         w = 1/self.sig**2        
         n = self.N[i]
         zj = self.weights[i]    
         delta = [(x - shift, z) for (x, z) in zip(self.data, zj)  if(x > shift)]
        
         weightedLogDelta = sum([z*log(x) for (x, z) in delta])
         weightedInverseDelta = sum([z/x for (x, z) in delta])
         return -n*alpha*log(betta) + n*math.lgamma(alpha) + betta*weightedInverseDelta + (alpha + 1)*weightedLogDelta  + w*(alpha - self.al0)**2/2 




    
    
        
def igmm(data, mode, z, z_tol = 0.1, max_iter = 100, x_tol = 0.005):
    last_loglike = float('inf')
    mix = InverseGammaMixture(data, mode, z, z_tol)
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

