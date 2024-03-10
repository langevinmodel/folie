"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from abc import ABC, abstractmethod
from .._numpy import np
from typing import Union


class TransitionDensity(ABC):
    use_jac = False

    def __init__(self, model):
        """
        Class which represents the transition density for a model, and implements a __call__ method to evalute the
        transition density (bound to the model)

            Parameters
            ----------
            model: the SDE model, referenced during calls to the transition density
        """
        self._model = model
        self._min_prob = np.log(1e-30)  # used to floor probabilities when evaluating the log

        # TODO: Trouver un moyen de set la dimensionalité du système à partir des données
        if self._model.dim <= 1:
            self._logdensity = self._logdensity1D
        else:
            if not hasattr(self, "_logdensityND"):
                raise ValueError("This transition density does not support multidimensionnal model.")
            self._logdensity = self._logdensityND

    @property
    def model(self):
        """Access to the underlying model"""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def check_dim(self, dim, **kwargs):
        """
        Set correct dimension for latter evaluation
        """

    def preprocess_traj(self, trj, **kwargs):
        """
        Basic preprocessing
        """
        if "xt" not in trj:  # ie, not preprocessing yet
            trj["xt"] = trj["x"][1:]
            trj["x"] = trj["x"][:-1]
            if "bias" in trj:
                trj["bias"] = trj["bias"][:-1]
            else:
                trj["bias"] = np.zeros((1, trj["x"].shape[1]))
            if hasattr(self._model, "dim_h"):
                if self._model.dim_h > 0:
                    trj["sig_h"] = np.zeros((trj["x"].shape[0], 2 * self._model.dim_h, 2 * self._model.dim_h))
                    trj["x"] = np.concatenate((trj["x"], np.zeros((trj["x"].shape[0], self._model.dim_h))), axis=1)
                    trj["xt"] = np.concatenate((trj["xt"], np.zeros((trj["xt"].shape[0], self._model.dim_h))), axis=1)
                    trj["bias"] = np.concatenate((trj["bias"], np.zeros((trj["bias"].shape[0], self._model.dim_h))), axis=1)
            trj = self.model.preprocess_traj(trj, **kwargs)
        return trj

    def density(self, x0: Union[float, np.ndarray], xt: Union[float, np.ndarray], dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(self._logdensity(x0, xt, dt))

    def __call__(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        return (-np.sum(np.maximum(self._min_prob, self._logdensity(**trj))) / weight,)
