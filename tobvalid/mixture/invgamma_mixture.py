"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import numpy as np
import scipy.stats as st
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import scipy as sp
import os
import sys
import seaborn as sns
import pandas as pd
from matplotlib import ticker
from scipy.optimize import minimize_scalar
from ._base import BaseMixture
from ._report import Report


class InverseGammaMixture(BaseMixture):
    def __init__(self, n_modes=1, tol=1e-5, max_iter=1000, ext="classic"):
        
        if(n_modes == 'auto'):
            n_modes = 1

        self._extenstions = ["classic", "stochastic"]
        self._ext = ext
        BaseMixture.__init__(self, n_modes, tol, max_iter)
        self._file_ext = "_sigd"
        self._xlabel = "Atomic B values"


    def _check_initial_custom_parameters(self, **kwargs):
        if not (self._ext in  self._extenstions) :
            raise ValueError("Unsupported EM extension: {}.".format(self._ext))

    def _check_parameters(self, X, **kwargs):
        if self._converged == False and not("z" in kwargs) and self.n_modes > 1:
            raise ValueError("Inverse Gamma Mixture requires 'z'")

    def _init_parameters(self, **kwargs):
        self.c = 0.1
        self.al0 = np.array([3.5] + [10]*(self.n_modes - 1))
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

        res = minimize_scalar(lambda x: self.__loglike(x, ss), bounds=(0, 1.2), method='bounded')
        step = res.x
  
        conv = False
        j = np.random.randint(self.n_modes)
        while not conv:
            
            if self._ext == "classic":
                self.alpha = np.maximum(
                    ialpha + step*ss[0], np.array([0.1]*self.n_modes))
                self.betta = np.maximum(
                    ibetta + step*ss[1], np.array([0.1]*self.n_modes))
                self.shift = ishift + step*ss[2]
            elif self._ext == "stochastic":
                self.alpha[j] = np.maximum(
                    ialpha[j] + step*ss[0][j], np.array([0.1]*self.n_modes)[j])
                self.betta[j] = np.maximum(
                    ibetta[j] + step*ss[1][j], np.array([0.1]*self.n_modes)[j])
                self.shift[j] = ishift[j] + step*ss[2][j]
            else:
                raise ValueError("Unsupported EM extension: {}.".format(self._ext))

            loglike = self.__loglike(1, np.zeros((3, self.n_modes)))
            if loglike > ifvalue:
                step = step*0.5
            else:
                conv = True
                self._loglike = loglike
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
        MinB = np.round(np.amin(self.data), 3)
        MaxB = np.round(np.amax(self.data), 3)
        MeanB = np.round(np.mean(self.data), 3)
        MedB = np.round(np.median(self.data), 3)
        VarB = np.round(np.var(self.data), 3)
        skewB = np.round(st.skew(self.data), 3)
        kurtsB = np.round(st.kurtosis(self.data), 3)
        firstQ, thirdQ = np.round(np.percentile(self.data, [25, 75]), 3)

        return {"Atom numbers": [nB], "Minimum B value": [MinB], 'Maximum B value': [MaxB], 'Mean': [MeanB], 'Median': [MedB], 'Variance': [VarB], 'Skewness': [skewB],
                'Kurtosis': [kurtsB], 'First quartile': [firstQ], 'Third quartile': [thirdQ]}

    def albeplot(self, plt, title='Alpha-Beta Plot'):

        fig, ax = plt.subplots()
        for i in range(self.n_modes):
            ax.plot(self.alpha[i], np.sqrt(self.betta[i]), marker='o')

        d = os.path.dirname(sys.modules["tobvalid"].__file__)
        xx = np.load(os.path.join(d, "templates/xx.npy"))
        yy = np.load(os.path.join(d, "templates/yy.npy"))
        kde = np.load(os.path.join(d, "templates/albe_kde.npy"))
        
        N = 30
        locator = ticker.MaxNLocator(N + 1, min_n_ticks=N)
        lev = locator.tick_values(kde.min(), kde.max())

        cfset = ax.contourf(xx, yy, kde, cmap='Reds', levels=lev[1:])
        fig.colorbar(cfset)

        
        y_max = max(max(self.alpha.max()*3, np.sqrt(self.betta).max()) + 3, 45)
        x_max = y_max // 3
        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\sqrt{\beta}$')
        plt.title(title)

    def clusterplot(self, plt, title="Clusters"):

        if self.n_modes == 1:
            return
        plt.figure()
        clusters = self.clusters()
        x = np.linspace(start=min(self.data), stop=max(self.data), num=1000)
        clust_num = np.concatenate(
            (np.zeros(len(clusters[0])), np.ones(len(clusters[1]))),  axis=None)
        cl_values = np.concatenate((clusters[0], clusters[1]),  axis=None)

        for i in np.arange(2, self.n_modes):
            clust_num = np.concatenate(
                (clust_num, np.full(len(clusters[i]), i)),  axis=None)
            cl_values = np.concatenate((cl_values, clusters[i]),  axis=None)

        df = pd.DataFrame({'BValues': cl_values, 'Clusters': clust_num})
        counts, bins = np.histogram(cl_values, bins=np.histogram(
            self.data, bins='scott', density=True)[1], density=True)

        groups = df.groupby([pd.cut(df.BValues, bins)])

        df_clust_per = ((groups.Clusters.sum()/groups.Clusters.count())).values

        norm = cl.Normalize(np.nanmin(df_clust_per),
                            np.nanmax(df_clust_per))
        colors = plt.cm.get_cmap(plt.get_cmap('seismic'))(norm(df_clust_per))
        plt.bar(bins[:-1], counts, width=(bins[1:] - bins[:-1]),
                align="edge", color=colors)

        values = self.pdf(x)

        plt.plot(x, values)

        plt.plot(x, values, color='black')
        plt.xlabel(self._xlabel)
        plt.ylabel("Density")
        plt.title(title)

    def __loglike(self, a, grad):
        w = np.array([1/self.sig**2] + [1/self.sig]*(self.n_modes - 1))

        weightedLogDelta = np.zeros((self.n_modes))
        weightedInverseDelta = np.zeros((self.n_modes))
        inverseSquareDelta = np.zeros((self.n_modes))

        for i in range(self.n_modes):

            n = self.N[i]

            zj = self.Z.T[i]
            idx = self.data > self.shift[i] + a*grad[2][i]
            delta = self.data[idx] - self.shift[i] - a*grad[2][i]

            zj = zj[idx]


            weightedLogDelta[i] = np.dot(zj, np.log(delta))
            weightedInverseDelta[i] = np.sum(zj/delta)
            inverseSquareDelta[i] = np.sum(zj/delta**2)


        alpha = np.maximum(self.alpha + a*grad[0], np.array([0.1]*self.n_modes))
        betta = np.maximum(self.betta + a*grad[1], np.array([0.1]*self.n_modes))


        return np.dot(-self.N*alpha*np.log(betta) + self.N*special.gammaln(alpha) + betta*
                               weightedInverseDelta + (alpha + 1)*weightedLogDelta + w*(alpha - self.al0)**2/2, self.mix)

    def CalcFisherMatrix(self):
        w = np.array([1/self.sig**2] + [1/self.sig]*(self.n_modes - 1))

        
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

            w1 = w[i]
            d2f[i] = np.array([
                [n*special.polygamma(1, alpha) + w1, -
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

        inv = sp.linalg.pinvh(self.__d2l)
        grad = np.transpose(np.array([self.__gradAlpha, self.__gradBetta, self.__gradB])).flatten()
        shiftc = np.matmul(-inv, grad)
        return shiftc.reshape((self.n_modes, 3)).T

    def params(self):
        return {"mix": self.mix, "alpha": self.alpha, "betta": self.betta, "shift": self.shift}

    def report(self, filename):

        report = Report(
            "Expecation Maximization of Inverse Gamma Mixture Model")
        report.head("Input")
        report.vtable(["Parameter", "Value", "Default Value"], [["File", filename, ""],
                                                                ["EM extension", self._ext, "classic"],
                                                                ["Number of modes",
                                                                    self.n_modes, 1],
                                                                ["Tolerance",
                                                                    self.tol, 1e-04],
                                                                ["Maximum Iterations",
                                                                    self.max_iter, 1000],
                                                                ["Number of Iterations", self.nit, 0],
                                                                ["Time in seconds", np.round(self.time(), 2), 0]])

        report.head("Output")

        report.htable(["Distribution"] + list(range(1, self.n_modes + 1)),
                      {'Mix parameters': np.round(self.mix, 3).tolist(), 'alpha': np.round(self.alpha, 3).tolist(), 'beta': np.round(self.betta, 3).tolist(), 'shift': np.round(self.shift, 3).tolist()})

        report.head('Parameters of B value distribution')
        report.htable(["",  ""],
                      self.__statistics__()
                      )

        if (self.n_modes > 1):

            clusters = self.clusters()
            report.head("")
            report.htable(["Clusters"] + list(range(1, self.n_modes + 1)),
                          {'mean': [np.round(np.mean(cl), 3) for cl in clusters], 'std': [np.round(np.std(cl), 3) for cl in clusters]})

        report.head("Plots")

        title = "Inverse Gamma Mixture: {}" if self.n_modes > 1 else "Inverse Gamma Distribution: {}"
        report.image(plt, self.mixtureplot, filename + ".mixture" +
                     self._file_ext, title.format(filename))
        if(self.n_modes > 1):
            report.image(plt, self.clusterplot, filename +
                         ".clusters" + self._file_ext, "Clusters: {}".format(filename))
        report.image(plt, self.probplot, filename + ".pp" +
                     self._file_ext, "P-P Plot: {}".format(filename))
        report.image(plt, self.qqplot, filename + ".qq" +
                     self._file_ext, "Q-Q Plot: {}".format(filename))

        report.image(plt, self.albeplot, filename + ".albe" +
                     self._file_ext, "'Alpha-Beta Plot': {}".format(filename))

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
