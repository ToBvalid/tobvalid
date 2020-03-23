# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:12:19 2019

@author: KavehB
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as special
import seaborn as sns

from ._base import BaseMixture
from ._html_generator import HTMLReport
from ._json_generator import JSONReport
from ._report import Report


class GaussianMixture(BaseMixture):
    def __init__(self, n_modes=1, tol=1e-3, max_iter=100):
        BaseMixture.__init__(self, n_modes, tol, max_iter)

    def _check_initial_custom_parameters(self, **kwargs):
        return

    def _check_parameters(self, X, **kwargs):
        return

    def _init_parameters(self, **kwargs):
        mu_min = np.min(self.data)
        mu_max = np.max(self.data)
        self.mu = np.array([mu_min + (mu_max - mu_min)*(2.*i + 1) /
                            (2.*self.n_modes) for i in range(self.n_modes)])
        self.sigma = np.ones(self.n_modes)*(mu_max-mu_min) / \
            (self.n_modes*np.sqrt(12.))

    def _m_step(self):
        N = np.sum(self.Z, axis=0)
        self.mu = np.sum((self.Z * self.data_n) *
                         np.reciprocal(N, dtype=float), axis=0)

        diff = self.data_n - self.mu
        diffSquare = diff*diff
        wdiffSquare = diffSquare*self.Z
        self.sigma = np.sqrt(
            np.sum(wdiffSquare*np.reciprocal(N, dtype=float), axis=0))

        self.mix = N*(1./np.sum(N))

        wp = self._mix_values()
        self._loglike = -np.sum(np.log(wp[wp > 1e-07]))
        return True

    def params(self):
        return {"mix": self.mix, "mu": self.mu, "sigma":self.sigma}

    def _pdf(self, X):
        u = (X - self.mu) / np.abs(self.sigma)
        y = (1 / (np.sqrt(2 * np.pi) * np.abs(self.sigma))) * np.exp(-u * u / 2)
        return y

    def _cdf(self, X):
        return (1.0 + special.erf((X - self.mu) / (np.sqrt(2.0)*self.sigma))) / 2.0

    def _ppf(self, p):
        return self.mu + self.sigma*np.sqrt(2.0)*special.erfinv(2*p - 1)

    def report(self, filename):

        report = Report("Expecation Maximization of Gaussian Mixture Model")
        report.head("Input")
        report.vtable(["Parameter", "Value", "Default Value"], [["File", filename, ""],
                                                                ["Number of modes", self.n_modes, 1], ["Tolerance", self.tol, 1e-3], ["Maximum Iterations", self.max_iter, 100]])

        report.head("Output")

        report.htable(["Distribution"] + list(range(1, self.n_modes + 1)),
                      {'Mix parameters': self.mix.tolist(), 'Mu': self.mu.tolist(), 'Sigma': self.sigma.tolist()})
        report.head("Plots")

        x = np.linspace(start=min(self.data), stop=max(self.data), num=1000)
        x = np.unique(self.data)
        x.sort()

        plt.figure()
        sns.set_style("white")
        sns.distplot(self.data, bins=30, kde=False, norm_hist=True)
        values = self.pdf(x)

        plt.plot(x, values, label="mixture")
        plt.legend()
        plt.title(filename)
        report.image(plt, filename)

        return report

    def savehtml(self, path, filename):
        report = self.report(filename)
        htmlreport = HTMLReport()
        htmlreport.save(report, path, filename)

    def savejson(self, path, filename):
        report = self.report(filename)
        jsonreport = JSONReport()
        jsonreport.save(report, path, filename)
