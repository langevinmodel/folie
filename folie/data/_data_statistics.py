from collections import namedtuple
from .._numpy import np
import scipy.stats
import scipy.optimize

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
    return DescribeResult(nobs, dim, np.asarray(X.min(0)), np.asarray(X.max(0)), np.asarray(X.mean(0)), np.asarray(X.var(0)))


def sum_stats(d1, d2):
    return DescribeResult(
        d1.nobs + d2.nobs,
        d1.dim,
        np.minimum(d1.min, d2.min),
        np.maximum(d1.max, d2.max),
        (d1.mean * d1.nobs + d2.mean * d2.nobs) / (d1.nobs + d2.nobs),
        ((d1.variance + d1.mean**2) * d1.nobs + (d2.variance + d2.mean**2) * d2.nobs) / (d1.nobs + d2.nobs) - ((d1.mean * d1.nobs + d2.mean * d2.nobs) / (d1.nobs + d2.nobs)) ** 2,
    )


def domain(stats, Npoints=75):
    """
    Build an array that is representative of the domain of the data points
    """
    return np.linspace(stats.min, stats.max, Npoints)


def _beta_params_from_mean_var(mean, var, uniform_points, loc=0, scale=1, optimize=True):
    # Define the objective function to minimize
    def objective(x):
        a, b = x
        beta = scipy.stats.beta.ppf(uniform_points, a, b, loc=loc, scale=scale)
        return (beta.mean() - mean) ** 2 + (beta.var() - var) ** 2

    m = (mean - loc) / scale
    v = var / scale**2
    # Initial guess for parameters
    initial_guess = [m * (m * (1 - m) / v - 1.0), (1 - m) * (m * (1 - m) / v - 1.0)]
    if not optimize:
        return initial_guess

    # Constraints: a, b > 0
    constraints = [{"type": "ineq", "fun": lambda x: x[0]}, {"type": "ineq", "fun": lambda x: x[1]}]

    # Minimize the objective function
    result = scipy.optimize.minimize(objective, initial_guess, constraints=constraints)

    return result.x


def representative_array(stats, Npoints=75, optimize=False):
    """
    Build an array with the same statistics than stats with Npoints.
    This is an helper function to fit functions with a reduced number of points
    If optimize is True, then the parameters are ajusted to match the statistics otherwise this is an approximation
    """
    uniform = np.linspace(np.zeros_like(stats.min), np.ones_like(stats.max), Npoints)
    rep_array = np.empty_like(uniform)
    scale = stats.max - stats.min

    for d in range(stats.dim):
        a, b = _beta_params_from_mean_var(stats.mean[d], stats.variance[d], uniform[:, d], loc=stats.min[d], scale=scale[d], optimize=optimize)
        rep_array[:, d] = scipy.stats.beta.ppf(uniform[:, d], a, b, loc=stats.min[d], scale=scale[d])
    return rep_array
