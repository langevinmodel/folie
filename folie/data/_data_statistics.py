from collections import namedtuple
import numpy as np
import scipy.stats

DescribeResult = namedtuple("DescribeResult", ("nobs", "dim", "min", "max", "mean", "variance"))


def traj_stats(X):
    """
    Simply return the dimension of the data
    """
    if X.ndim == 2:
        nobs, dim = X.shape
    elif X.ndim == 1:
        nobs = X.shape[0]
        dim = 1
        X = X.reshape(-1, 1)
    else:
        nobs = X.shape[0]
        dim = X.shape[1]
    return DescribeResult(nobs, dim, X.min(axis=0), X.max(axis=0), X.mean(axis=0), X.var(axis=0))


def sum_stats(d1, d2):
    return DescribeResult(d1.nobs + d2.nobs, d1.dim, np.minimum(d1.min, d2.min), np.maximum(d1.max, d2.max), (d1.mean * d1.nobs + d2.mean * d2.nobs) / (d1.nobs + d2.nobs), d1.variance)


def domain(stats, Npoints=75):
    """
    Build an array that is representative of the domain of the data points
    """

    return np.linspace(stats.min, stats.max, Npoints)


def representative_array(stats, Npoints=75):
    """
    Build an array with the same statistics than stats with Npoints.
    This is an helper function to fit functions with a reduced number of points
    """
    uniform = np.linspace(np.zeros_like(stats.min), np.ones_like(stats.max), Npoints)
    rep_array = np.empty_like(uniform)
    scale = stats.max - stats.min
    for d in range(stats.dim):
        m = (stats.mean[d] - stats.min[d]) / scale[d]
        v = stats.variance[d] / scale[d] ** 2
        a = m * (m * (1 - m) / v - 1.0)
        b = (1 - m) * (m * (1 - m) / v - 1.0)
        rep_array[:, d] = scipy.stats.beta.ppf(uniform[:, d], a, b, loc=stats.min[d], scale=scale[d])
        # Il faudrait ensuite optimiser a et b pour avoir les bonnes moyennes et variances
    return rep_array
