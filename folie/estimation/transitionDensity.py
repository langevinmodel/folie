"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from abc import ABC, abstractmethod
import numpy as np
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
        trj["xt"] = trj["x"][1:]
        trj["x"] = trj["x"][:-1]
        if hasattr(self._model, "dim_h"):
            if self._model.dim_h > 0:
                trj["sig_h"] = np.zeros((trj["x"].shape[0], 2 * self._model.dim_h, 2 * self._model.dim_h))
                trj["x"] = np.concatenate((trj["x"], np.zeros((trj["x"].shape[0], self._model.dim_h))), axis=1)
                trj["xt"] = np.concatenate((trj["xt"], np.zeros((trj["xt"].shape[0], self._model.dim_h))), axis=1)
        return trj

    def density(self, x0: Union[float, np.ndarray], xt: Union[float, np.ndarray], dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(self._logdensity(x0, xt, dt))

    @abstractmethod
    def _logdensity1D(self, x0: Union[float, np.ndarray], xt: Union[float, np.ndarray], dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        The transition density evaluated at these arguments
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        raise NotImplementedError

    def __call__(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        return (-np.sum(np.maximum(self._min_prob, self._logdensity(x0=trj["x"], xt=trj["xt"], dt=trj["dt"]))) / weight,)


class GaussianTransitionDensity(TransitionDensity):
    """
    Class that represent Gaussian transition density
    """

    def _logdensity1D(self, x0: Union[float, np.ndarray], xt: Union[float, np.ndarray], dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        The transition density evaluated at these arguments
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        E = self._mean(x0, dt).ravel()
        V = self._variance(x0, dt).ravel()
        return -0.5 * ((xt.ravel() - E) ** 2 / V) - 0.5 * np.log(np.sqrt(2 * np.pi) * V)

    def _logdensityND(self, x0, xt, dt):
        """
        The transition density evaluated at these arguments
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        E = self._mean(x0, dt)
        V = self._variance(x0, dt)
        return -0.5 * np.dot(np.dot(xt - E, np.linalg.inv(V)), xt - E) - 0.5 * np.log(np.sqrt(2 * np.pi) * np.linalg.det(V))

    def compute_noise(self, trj, coefficients):
        """
        Allow to estimate the noise from a trajectories and a fitted model
        TODO: En vrai, c'est globalement ce qui est calculé dans chaque log density (qd elle sont gaussiennes), es-ce qu'on peut le réutiliser?
        """
        E = self._mean(trj["x"], 0, trj["dt"])
        V = self._variance(trj["x"], 0, trj["dt"])
        return (trj["xt"].ravel() - E) ** 2 / V

    def run_step(self, x, dt, dW, t=0.0):
        E = self._mean(x, 0, dt)
        V = self._variance(x, 0, dt)
        return x + E + np.sqrt(V) * dW

    @abstractmethod
    def _mean(self, x, t, dt):
        raise NotImplementedError

    @abstractmethod
    def _variance(self, x, t, dt):
        raise NotImplementedError
