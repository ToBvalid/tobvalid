# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:12:19 2019

@author: KavehB
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from .base import BaseMixture

class GaussianMixture(BaseMixture):
     def __init__(self, n_modes = 1, tol = 1e-3, max_iter = 100):
        BaseMixture.__init__(self, n_modes, tol, max_iter) 
     
     def _check_parameters(self, X, **kwargs):
        return
     
     def _init_parameters(self, **kwargs):
        mu_min = np.min(self.data)
        mu_max = np.max(self.data)
        self.mu = np.array([mu_min +  (mu_max - mu_min)*(2.*i + 1)/(2.*self.n_modes) for i in range(self.n_modes)])
        self.sigma =   np.ones(self.n_modes)*(mu_max-mu_min)/(self.n_modes*np.sqrt(12.))

        
     def _m_step(self):
        N = np.sum(self.Z, axis = 0)

        self.mu = np.sum((self.Z* self.data_n) *np.reciprocal(N, dtype=float), axis = 0 )

        diff = self.data_n - self.mu  
        diffSquare = diff*diff
        wdiffSquare = diffSquare*self.Z
        self.sigma = np.sqrt(np.sum(wdiffSquare*np.reciprocal(N, dtype=float), axis = 0 ))

        self.mix = N*(1./np.sum(N))
        
        wp = self._mix_values()
        self._loglike = -np.sum(np.log(wp[wp > 1e-07]))
        return True
        
     def _pdf(self, X):
         u = (X - self.mu) / np.abs(self.sigma)
         y = (1 / (np.sqrt(2 * np.pi) * np.abs(self.sigma))) * np.exp(-u * u / 2)
         return y
     
     def reportHTML(self, path, filename):
        from tobevalid.tools.html_generator import HTMLReport
        
        report = HTMLReport()
        report.open(path)
        report.head1("Expecation Maximization of Gaussian Mixture Model").head2("Input")
        report.vtable(["Parameter","Value","Default Value"], [["File", filename, ""], 
                       ["Number of modes", self.n_modes, 1], ["Tolerance", self.tol , 1e-3],["Maximum Iterations", self.max_iter, 100]])
        report.head2("Output")
        report.htable(["Distribution","1","2"], {'Mix parameters':self.mix, 'Mu':self.mu, 'Sigma':self.sigma})
        report.head2("Plot")
        
        x = np.linspace(start=min(self.data), stop=max(self.data), num=1000)
        x = np.unique(self.data)
        x.sort() 
        
        sns.set_style("white")
        sns.distplot(self.data, bins=30, kde=False, norm_hist=True)
        values = self.pdf(x)
        plt.plot(x, values, label="mixture")
        plt.legend()
        plt.title(filename) 
        
        report.image(plt, filename + ".png")
        report.close().save(path + "/" + filename + ".html")
     
     @staticmethod
     def search(X, tol = 1e-3, max_iter = 100, mod_tol = 0.2, mix_tol = 0.1, peak_tol = 0.9, ret_mix = False):
        lastres = None
        mode = 2
        while (True):
            mixture = GaussianMixture(mode, tol, max_iter)
            mixture.fit(X)
            z = mixture.Z.T
             
            if ((mixture.mix < mix_tol).any()):
                break
            if(min(z.sum(axis = 1)*mixture.mix/X.size) < 0.1):
                break
            it = 0
            for i in np.arange(mode - 1):
                z1 = z[i].sum()
                z2 = z[i + 1].sum()
                z12 = z[i, z[i] <= z[i + 1]].sum() + z[i + 1, z[i] >= z[i + 1]].sum()
                if (2*z12/(z1 + z2) >= mod_tol):
                    break
                peak1 = mixture.mix[i]/(np.sqrt(2*np.pi)*mixture.sigma[i])
                peak2 = mixture.mix[i + 1]/(np.sqrt(2*np.pi)*mixture.sigma[i + 1])
                if(abs(mixture.mu[i + 1] - mixture.mu[i]) < (mixture.sigma[i + 1] + mixture.sigma[i]) 
                    and min(peak1, peak2)/max(peak1, peak2) > peak_tol):
                    break
                it += 1
            if(it < mode - 1):
                break
            lastres = mixture
            mode += 1
        if(ret_mix == False):
            return mode - 1
        if( mode == 2):
            mixture = GaussianMixture(1, tol, max_iter)
            mixture.fit(X)
            return (1, mixture)
        return (mode - 1, lastres)
  
    
     
    
