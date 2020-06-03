"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"
Institute of Molecular Biology and Biotechnology (IMBB)
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import numpy as np
import seaborn as sns
from scipy.optimize import root_scalar
from ._html_generator import HTMLReport
from ._json_generator import JSONReport
import statsmodels.api as sm
from ..stats import kde_silverman


class BaseMixture:
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    def __init__(self, n_modes, tol, max_iter, **kwargs):
        self._converged = False
        self.n_modes = n_modes
        self._fit = False

        if n_modes == 'auto':
            self.n_modes = 1
            self._fit = True

        self.tol = tol
        self.max_iter = max_iter
        self._loglike = 0
        self.mix = np.ones(self.n_modes)/self.n_modes
        self.nit = 0
        self._ext = "mm"
        self._check_initial_parameters(**kwargs)

    def _check_initial_parameters(self, **kwargs):
        """Check values of the basic parameters.

        """
        if self.n_modes < 1:
            raise ValueError("Invalid value for 'n_modes': %d "
                             "EM requires at least one mode"
                             % self.n_modes)

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        # Check all the parameters values of the derived class
        self._check_initial_custom_parameters(**kwargs)

    def fit(self, X, **kwargs):
        self._check_X(X)
        self._check_parameters(X, **kwargs)

        self.data = X
        if (self._fit == True):
            modes, kernel = kde_silverman(self.data)
            self.n_modes = modes[0]
            self._kernel = kernel
            self._modes = modes[1]
            self.mix = np.ones(self.n_modes)/self.n_modes

        self.data_n = np.repeat(
            self.data[np.newaxis, ...], self.n_modes, axis=0).T

        self._init_parameters(**kwargs)

        self._converged_ = False

        lower_bound = -np.infty
        self.nit = 0

        for n_iter in np.arange(1, self.max_iter + 1):

            prev_lower_bound = lower_bound
            self._e_step()
            if not self._m_step():
                break

            lower_bound = self.loglike()

            change = lower_bound - prev_lower_bound

            if n_iter > 1 and np.abs(change/prev_lower_bound) < self.tol:
                self._converged_ = True
                break
        self.nit = n_iter
        self._converged = True
        self.__ppplot = sm.ProbPlot(self.data, self, fit=False)

    def loglike(self):
        return self._loglike

    def pdf(self, X):
        if isinstance(X, (np.ndarray)):
            return np.matmul(self._pdf(np.repeat(X[np.newaxis, ...], self.n_modes, axis=0).T), self.mix.T)

        if isinstance(X, (int, float)):
            return np.matmul(self._pdf(X), self.mix.T)

        if isinstance(X, (list, tuple)) and all(isinstance(x, (int, float)) for x in X):
            return self.pdf(np.array(X))

        raise ValueError(
            "Expected numerical array or number, got {} instead.".format(X))

    def cdf(self, X):
        if isinstance(X, (np.ndarray)):
            return np.matmul(self._cdf(np.repeat(X[np.newaxis, ...], self.n_modes, axis=0).T), self.mix.T)

        if isinstance(X, (int, float)):
            return np.matmul(self._cdf(X), self.mix.T)

        if isinstance(X, (list, tuple)) and all(isinstance(x, (int, float)) for x in X):
            return self.cdf(np.array(X))

        raise ValueError(
            "Expected numerical array or number, got {} instead.".format(X))

    def ppf(self, p):

        if self.n_modes == 1:
            return self._ppf(p)

        if isinstance(p, (np.ndarray)) or (isinstance(p, (list, tuple)) and all(isinstance(x, (int, float)) for x in p)):
            return np.array([self.ppf(pi) for pi in p])

        ppf = self._ppf(p)
        low = np.min(ppf)
        high = np.max(ppf)
        return root_scalar(lambda x: self.cdf(x) - p, method='brentq', bracket=[low, high], maxiter=10).root

    def mixpdf(self, X):
        if isinstance(X, (np.ndarray)):
            return self._pdf(np.repeat(X[np.newaxis, ...], self.n_modes, axis=0).T)

        if isinstance(X, (int, float)):
            return self._pdf(X)

        if isinstance(X, (list, tuple)) and all(isinstance(x, (int, float)) for x in X):
            return self.mixpdf(np.array(X))

        raise ValueError(
            "Expected numerical array or number, got {} instead.".format(X))

    def clusters(self):
        result = []
        for i in np.arange(self.n_modes):
            result.append([])

        for d, z in zip(self.data, self.Z):
            result[np.random.choice(np.arange(self.n_modes), 1, p=z)[
                0]].append(d)

        return result

    def mixtureplot(self, plt, title="Mixture"):

        x = np.linspace(start=min(self.data), stop=max(self.data), num=1000)

        plt.figure()
        sns.set_style("white")
        sns.distplot(self.data, bins='scott', kde=False, hist_kws=dict(
            edgecolor=None, linewidth=0, color='grey'), norm_hist=True)
        values = self.pdf(x)

        plt.plot(x, values, label="mixture", color='black')
        plt.legend()
        plt.title(title)

    def probplot(self, plt, title='P-P Plot'):
        plt.figure()
        self.__ppplot.ppplot(line='45', color='blue')
        plt.title(title)

    def qqplot(self, plt, title='Q-Q Plot'):

        x = np.sort(self.data)
        n = x.size
        y = np.arange(1, n+1) / n
        if(n > 200):
            k = n//200
        else:
            k = 1
        x1 = self.ppf(y[:-1:k])
        plt.figure()

        line = np.linspace(min(self.data), max(self.data), 100)
        plt.plot(x1, x[:-1:k], linewidth=6, color='blue')
        plt.plot(line, line, "r-")
        plt.xlim(min(self.data), max(self.data))
        plt.ylim(min(self.data), max(self.data))
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.title(title)

    def modeplot(self, plt, title="Modes"):

        x = np.linspace(start=min(self.data), stop=max(self.data), num=1000)

        plt.figure()
        sns.set_style("white")
        sns.distplot(self.data, bins='scott', kde=False, hist_kws=dict(
            edgecolor=None, linewidth=0, color='grey'), norm_hist=True)
        values = self._kernel(x)

        plt.plot(x, values, color='black')
        for mode in self._modes:
            plt.plot(mode, self._kernel(mode), marker="*", label=mode)

        plt.legend()
        plt.title(title)

    def savehtml(self, path, filename, dpi=None):
        report = self.report(filename)
        htmlreport = HTMLReport(dpi)
        htmlreport.save(report, path, filename + self._ext)

    def savejson(self, path, filename, dpi=None):
        report = self.report(filename)
        jsonreport = JSONReport(dpi)
        jsonreport.save(report, path, filename)

    def report(self, filename, dpi=None):
        pass

    def params(self):
        pass

    def _check_initial_custom_parameters(self, **kwargs):
        pass

    def _pdf(self, X):
        pass

    def _cdf(self, X):
        pass

    def _ppf(self, p):
        pass

    def _check_parameters(self, X, **kwargs):
        pass

    def _check_X(self, X):

        if not isinstance(X, (np.ndarray)):
            raise ValueError("Expected ndarray, got {} instead.".format(X))

        if len(X.shape) != 1:
            raise ValueError(
                "Expected one-dimentional ndarray, got {} instead.".format(X))

        if not X.dtype in [np.float32, np.float64]:
            raise ValueError(
                "Expected numerical ndarray, got {} instead.".format(X))

    def _init_parameters(self, **kwargs):
        pass

    def _mix_values(self):
        values = self._pdf(self.data_n)
        return values*self.mix

    def _calc_posterior_z(self):
        wp = self._mix_values()

        den = np.sum(wp, axis=1)

        for i in np.arange(len(wp)):
            for j in np.arange(len(wp[i])):
                if wp[i][j] != 0:
                    wp[i][j] = wp[i][j]/den[i]
        self.Z = wp

    def _e_step(self):
        self._calc_posterior_z()
        self.N = np.sum(self.Z.T, axis=1)
        self.mix = self.N*(1/np.sum(self.N))

    def _m_step(self):
        pass
