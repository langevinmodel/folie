from collections import namedtuple
import numpy as np

DescribeResult = namedtuple("DescribeResult", ("nobs", "min", "max", "mean", "variance"))


def traj_stats(X):
    """
    Simply return the dimension of the data
    """
    nobs, dim = X.shape
    return DescribeResult(nobs, np.min(X, axis=0), np.max(X, axis=0), np.mean(X, axis=0), np.var(X, axis=0))


def sum_stats(d1, d2):
    return DescribeResult(d1.nobs + d2.nobs, np.minimum(d1.min, d2.min), np.maximum(d1.max, d2.max), (d1.mean * d1.nobs + d2.mean * d2.nobs) / (d1.nobs + d2.nobs), d1.variance)


# Il faudrait faire un calcul de descripteurs avec accumulation sur les trajs ?


# TODO: Add some function that allow to take a list of keywords arguments and make a describe from it, so we can deal with various type of input into the fit function
