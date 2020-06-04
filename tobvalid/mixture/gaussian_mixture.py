"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as special


from ._base import BaseMixture
from ._report import Report


class GaussianMixture(BaseMixture):
    def __init__(self, n_modes=1, tol=1e-05, max_iter=100, **kwargs):
        BaseMixture.__init__(self, n_modes, tol, max_iter, **kwargs)
        self._ext = "_gmm"

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
        return {"mix": self.mix, "mu": self.mu, "sigma": self.sigma}

    def _pdf(self, X):
        u = (X - self.mu) / np.abs(self.sigma)
        y = (1 / (np.sqrt(2 * np.pi) * np.abs(self.sigma))) * np.exp(-u * u / 2)
        return y

    def _cdf(self, X):
        return (1.0 + special.erf((X - self.mu) / (np.sqrt(2.0)*self.sigma))) / 2.0

    def _ppf(self, p):
        return self.mu + self.sigma*np.sqrt(2.0)*special.erfinv(2*p - 1)

    def report(self, filename):

        report = Report("Expectation Maximization of Gaussian Mixture Model")
        if (self._fit == True):
            report.head("Mode search (Silverman Method)")
            report.htable(["Mode"] + list(range(1, self.n_modes + 1)),
                          {' ': self._modes})

            report.image(plt, self.modeplot, filename + ".silverman" +
                         self._ext, "Modes: {}".format(filename))

        report.head("Input")
        report.vtable(["Parameter", "Value", "Default Value"], [["File", filename, ""],
                                                                ["Number of modes", self.n_modes, 1], ["Tolerance", self.tol, 1e-05], ["Maximum Iterations", self.max_iter, 100], ["Number of Iterations", self.nit, 0]])

        report.head("Output")

        report.htable(["Distribution"] + list(range(1, self.n_modes + 1)),
                      {'Mix parameters': np.round(self.mix, 3).tolist(), 'Mu': np.round(self.mu, 3).tolist(), 'Sigma': np.round(self.sigma, 3).tolist()})
        report.head("Plots")

        report.image(plt, self.mixtureplot, filename + ".mixture" +
                     self._ext, "Gaussian Mixture: {}".format(filename))

        return report
