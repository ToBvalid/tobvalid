# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:26:01 2019

@author: KavehB
"""
import numpy as np
import scipy.stats as st
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
from matplotlib import  ticker

from ._base import BaseMixture
from ._report import Report


class InverseGammaMixture(BaseMixture):
    def __init__(self, n_modes=1, tol=1e-5, max_iter=100):

        if(n_modes == 'auto'):
           n_modes = 1
           
        BaseMixture.__init__(self, n_modes, tol, max_iter)
        self._ext = "_igmm"

    def _check_initial_custom_parameters(self, **kwargs):
        return

    def _check_parameters(self, X, **kwargs):
        if self._converged == False and not("z" in kwargs) and self.n_modes > 1:
            raise ValueError("Inverse Gamma Mixture requires 'z'")

    def _init_parameters(self, **kwargs):
        self.c = 0.1
        self.al0 = 3.5
        self.sig = 0.1
        self.epsilon = 1.0e-16
        self.step = 1

        if self.n_modes == 1:
            self.Z = np.ones(len(self.data)).reshape(len(self.data), 1)
        elif ("z" in kwargs):
            self.Z = kwargs["z"]

        N = self.Z.sum(axis=0)
        mdB = np.matmul(self.data, self.Z)/N
        minB = [np.min(self.data[self.Z[:, i] > self.c])
                for i in range(self.n_modes)]
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
        ifvalue = self.loglike()

        conv = False
        while not conv:
            self.alpha = np.maximum(
                ialpha + step*ss[0], np.array([0.1]*self.n_modes))
            self.betta = np.maximum(
                ibetta + step*ss[1], np.array([0.1]*self.n_modes))
            self.shift = ishift + step*ss[2]

            self.CalcFisherMatrix()

            if self._loglike > ifvalue:
                step = step*0.7
            else:
                conv = True
                break

            if step <= self.epsilon:
                self.alpha = ialpha.copy()
                self.betta = ibetta.copy()
                self.shiftv = ishift.copy()
                self._loglike = ifvalue
                break
        return conv

    def __statistics__(self):
        nB = len(self.data)
        MinB = np.amin(self.data)
        MaxB = np.amax(self.data)
        MeanB = np.mean(self.data)
        MedB = np.median(self.data)
        VarB = np.var(self.data)
        skewB = st.skew(self.data)
        kurtsB = st.kurtosis(self.data)
        firstQ, thirdQ = np.percentile(self.data, [25, 75])

        return {"Atom numbers": [nB], "Minimum B value": [MinB], 'Maximum B value': [MaxB], 'Mean': [MeanB], 'Median': [MedB], 'Variance': [VarB], 'Skewness': [skewB],
                'Kurtosis': [kurtsB], 'First quartile': [firstQ], 'Third quartile': [thirdQ]}


    def albeplot(self, plt, title='Alpha-Beta Plot'):

        if(self.n_modes > 1):
            return
        
   
        fig, ax = plt.subplots()
        for i in range(self.n_modes):
            ax.plot(self.alpha[i], np.sqrt(self.betta[i]), marker='o')

       
        d = os.path.dirname(sys.modules["tobevalid"].__file__)
        xx = np.load(os.path.join(d, "templates/xx.npy"))
        yy = np.load(os.path.join(d, "templates/yy.npy"))
        kde = np.load(os.path.join(d, "templates/albe_kde.npy"))

        N=30
        locator = ticker.MaxNLocator(N + 1, min_n_ticks=N)
        lev = locator.tick_values(kde.min(), kde.max())

        cfset = ax.contourf(xx, yy, kde, cmap='Reds', levels=lev[1:])
        cbar = fig.colorbar(cfset)

        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\sqrt{\beta}$')
        plt.title(title)


    def clusterplot(self, plt, title="Clusters"):

        if self.n_modes == 1:
            return
        fig = plt.figure()
        clusters = self.clusters()
        x = np.linspace(start=min(self.data), stop=max(self.data), num=1000)
        clust_num = np.concatenate(
            (np.zeros(len(clusters[0])), np.ones(len(clusters[1]))),  axis=None)
        cl_values = np.concatenate((clusters[0], clusters[1]),  axis=None)

        for i in np.arange(2, self.n_modes):
            clust_num = np.concatenate((clust_num, np.full(len(clusters[i]), i)),  axis=None)
            cl_values = np.concatenate((cl_values, clusters[i]),  axis=None)

        df = pd.DataFrame({'BValues': cl_values, 'Clusters': clust_num})
        counts, bins = np.histogram(cl_values, bins=np.histogram(self.data, bins='scott', density=True)[1], density=True)

        groups = df.groupby([pd.cut(df.BValues, bins)])
        df_bins = pd.cut(df.BValues, bins)
        df_clust_per = ((groups.Clusters.sum()/groups.Clusters.count())).values


        norm = cl.Normalize(np.nanmin(df_clust_per),
                             np.nanmax(df_clust_per))
        colors = plt.cm.get_cmap(plt.get_cmap('PRGn'))(norm(df_clust_per))
        plt.bar(bins[:-1], counts, width=(bins[1:] - bins[:-1]),
                align="edge", color=colors)

        values = self.pdf(x)
        plt.plot(x, values)

        plt.title(title)

    def CalcFisherMatrix(self):
        w = np.array([1/self.sig**2] + [0]*(self.n_modes - 1))

        self._loglike = 0

        self.__d2l = np.zeros((3*self.n_modes, 3*self.n_modes))
        self.__gradAlpha = [0]*self.n_modes
        self.__gradBetta = [0]*self.n_modes
        self.__gradB = [0]*self.n_modes

        d2f = [0]*self.n_modes

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

            d2f[i] = np.array([
                [n*special.polygamma(1, alpha) + w[0], -
                 n/betta, -n*alpha/betta],
                [-n/betta, n*alpha /
                    np.power(betta, 2), n*alpha*(alpha + 1)/np.power(betta, 2)],
                [-n*alpha/betta,  n*alpha*(alpha + 1)/np.power(betta, 2),  n*alpha*(alpha + 1)*(alpha + 3)/np.power(betta, 2)]])

        self.__gradAlpha = self.N*special.psi(self.alpha) - self.N*np.log(
            self.betta) + weightedLogDelta + w*(self.alpha - self.al0)
        self.__gradBetta = -self.N*self.alpha/self.betta + weightedInverseDelta
        self.__gradB = self.betta*inverseSquareDelta - \
            (self.alpha + 1)*weightedInverseDelta

        self._loglike = np.dot(-self.N*self.alpha*np.log(self.betta) + self.N*special.gammaln(self.alpha) + self.betta *
                               weightedInverseDelta + (self.alpha + 1)*weightedLogDelta + w*(self.alpha - self.al0)**2/2, self.mix)

        for i in range(self.n_modes):
            for j in range(3):
                for k in range(3):
                    self.__d2l[3*i + j, k + 3*i] = self.mix[i]*d2f[i][j][k]

    def shiftcalc(self, tol=1.0e-5):

        inv = np.linalg.inv(
            self.__d2l + tol*np.random.randn(3*self.n_modes, 3*self.n_modes))
        grad = np.transpose(
            np.array([self.__gradAlpha, self.__gradBetta, self.__gradB])).flatten()
        shiftc = np.matmul(-inv, grad)
        return shiftc.reshape((self.n_modes, 3)).T
        return True

    def params(self):
        return {"mix": self.mix, "alpha": self.alpha, "betta": self.betta, "shift": self.shift}

    def report(self, filename):

        report = Report(
            "Expecation Maximization of Inverse Gamma Mixture Model")
        report.head("Input")
        report.vtable(["Parameter", "Value", "Default Value"], [["File", filename, ""],
                                                                ["Number of modes",
                                                                    self.n_modes, 1],
                                                                ["Tolerance",
                                                                    self.tol, 1e-05],
                                                                ["Maximum Iterations", self.max_iter, 100]])

        report.head("Output")

        report.htable(["Distribution"] + list(range(1, self.n_modes + 1)),
                      {'Mix parameters': self.mix.tolist(), 'alpha': self.alpha.tolist(), 'beta': self.betta.tolist(), 'shift': self.shift.tolist()})

        report.head('Parameters of B value distribution')
        report.htable(["",  ""],
                      self.__statistics__()
                      )
        report.head("Plots")

        report.image(plt, self.mixtureplot, filename + ".mixture" +
                     self._ext, "Inverse Gamma Mixture: {}".format(filename))
        if(self.n_modes > 1):
            report.image(plt, self.clusterplot, filename +
                         ".clusters" + self._ext, "Clusters: {}".format(filename))
        report.image(plt, self.probplot, filename + ".pp" +
                     self._ext, "P-P Plot: {}".format(filename))
        report.image(plt, self.qqplot, filename + ".qq" +
                     self._ext, "Q-Q Plot: {}".format(filename))
                             
        if(self.n_modes == 1):
             report.image(plt, self.albeplot, filename + ".albe" +
                         self._ext, "'Alpha-Beta Plot': {}".format(filename))


        return report

    def _pdf(self, X):
        dist = st.invgamma(self.alpha, self.shift, self.betta)
        return dist.pdf(X)

    def _cdf(self, X):
        dist = st.invgamma(self.alpha, self.shift, self.betta)
        return dist.cdf(X)

    def _ppf(self, p):
        dist = st.invgamma(self.alpha, self.shift, self.betta)
        return dist.ppf(p)
