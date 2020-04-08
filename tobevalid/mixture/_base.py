# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:14:24 2019

@author: KavehB
"""

import numpy as np
import seaborn as sns
from scipy.optimize import root_scalar
from ._html_generator import HTMLReport
from ._json_generator import JSONReport


class BaseMixture:
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """
    
    def __init__(self, n_modes, tol, max_iter, **kwargs):
        self._fit = False
        self.n_modes = n_modes
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
        self._fit = True

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
        return root_scalar(lambda x: self.cdf(x) - p, method='bisect', bracket=[low, high], maxiter=10).root

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
        p = self.mixpdf(self.data).T
        p_sum = p.sum()
        lengths = p.sum(axis=1)/p_sum

        for i in np.arange(self.n_modes):
            result.append(np.random.choice(self.data.tolist(), int(
                len(self.data)*lengths[i]), p=p[i]/p[i].sum(), replace=False))
        return result

    def probplot(self, plt, title = 'P-P Plot'):
        q_range = range(0, 101)
        bins = np.percentile(self.data, q_range)
        counts, bins = np.histogram(self.data, bins)
        cum = np.cumsum(counts)
        cum = cum / max(cum)
        cdfs = self.cdf(bins)[1:]

        plt.plot(cdfs, cum, "o")
        min_value = np.floor(np.min([cdfs.min(), cum.min()]))
        max_value = np.ceil(np.max([cdfs.max(), cum.max()]))
        plt.plot([min_value, max_value], [min_value, max_value], 'r--')
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)

        plt.xlabel('Theoretical values')
        plt.ylabel('Observed values')
        plt.title(title)

    def mixtureplot(self, plt, title="Mixture"):

        x = np.linspace(start=min(self.data), stop=max(self.data), num=1000)
        # x = np.unique(self.data)
        # x.sort()

        plt.figure()
        sns.set_style("white")
        sns.distplot(self.data, bins='scott', kde=False, hist_kws=dict(edgecolor="k", linewidth=2), norm_hist=True)
        values = self.pdf(x)

        plt.plot(x, values, label="mixture")
        plt.legend()
        plt.title(title)

    def qqplot(self, plt, title = 'Q-Q Plot'):
        q_range = range(0, 101)
        bins = np.percentile(self.data, q_range)
        counts, bins = np.histogram(self.data, bins='scott')

        cum = np.cumsum(counts)
        cum = cum / max(cum)

        ppf_data = self.ppf(cum)
        plt.plot(ppf_data, bins[1:], "o")
        min_value = min(self.data)
        max_value = max(self.data)
        plt.plot([min_value, max_value], [min_value, max_value], 'r--')
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)

        plt.xlabel('Theoretical values')
        plt.ylabel('Observed values')
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
