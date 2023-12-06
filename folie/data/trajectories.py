from collections.abc import MutableSequence, Mapping
import numpy as np
from ._data_statistics import traj_stats, sum_stats


def Trajectory(x, dt):
    """
    Create dict_like object that encaspulate the trajectory data
    TODO: Use xarray DataSet?
    """
    return {"x": x, "dt": dt}


class Trajectories(MutableSequence):
    """
    Set of trajectories
    """

    def __init__(self, dt=None):
        self.dt = dt
        self.trajectories_data = []
        self.dim = None
        self.stats_data = None

    def _check_data(self, v):
        """ """
        if not isinstance(v, Mapping):
            v = Trajectory(v, self.dt)
        if len(v["x"].shape) == 1:
            dim_x = 1
        else:
            dim_x = v["x"].shape[-1]
        if self.dim is None:
            self.dim = dim_x
        elif self.dim != dim_x:
            raise ValueError("Inconsitent dimension between previously stored trajectory and currently added trajectory")
        if self.dt is None:
            self.dt = v["dt"]
        return v

    def __len__(self):
        return len(self.trajectories_data)

    def __getitem__(self, i):
        return self.trajectories_data[i]

    def __delitem__(self, i):
        del self.trajectories_data[i]

    def __setitem__(self, i, v):
        self.trajectories_data[i] = self._check_data(v)

    def insert(self, i, v):
        self.trajectories_data.insert(i, self._check_data(v))

    def __str__(self):
        return ["Trajectory of length {} and dimension {}.".format(len(trj), self.dim) for trj in self.trajectories_data]

    @property
    def stats(self):
        """
        Basic statistics on the data
        """
        if self.stats_data is None:
            self.stats_data = traj_stats(self.trajectories_data[0]["x"])
            for trj in self.trajectories_data[1:]:
                self.stats_data = sum_stats(self.stats_data, traj_stats(trj["x"]))
        return self.stats_data

    @property
    def nobs(self):
        return np.sum([len(trj) for trj in self.trajectories_data])

    @property
    def weights(self):
        return np.array([len(trj) for trj in self.trajectories_data])

    def to_xarray(self):
        """
        Return the data as a list of Dataset where extra variables are denoted by their name
        """
        raise NotImplementedError

    @classmethod
    def from_xarray(traj_list, data_key="x"):
        """Take as input a list of xarray Dataset"""
        raise NotImplementedError
