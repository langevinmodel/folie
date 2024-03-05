from .trajectories import *
from ._data_statistics import traj_stats, DescribeResult, representative_array
from .._numpy import np


def stats_from_input_data(X):
    if isinstance(X, Trajectories):
        return X.stats
    elif isinstance(X, DescribeResult):
        return X
    elif X is None:
        return DescribeResult(1, 1, np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    else:
        return traj_stats(X)
