"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union

# TODO: A faire, la réduction de la somme sur les pas de temps doit se faire dans la classe et si on construire des versions efficace on les appelle juste d'un nom différents comme CachedEulerDensity
# Les classes doivent aussi avoir une manière de rajouter des covariances sur les variables
# Ecrire tout ça en version ND


class TransitionDensity(ABC):
    def __init__(self, model):
        """
        Class which represents the transition density for a model, and implements a __call__ method to evalute the
        transition density (bound to the model)

        :param model: the SDE model, referenced during calls to the transition density
        """
        self._model = model
        self._min_prob = np.log(1e-30)  # used to floor probabilities when evaluating the log

    @property
    def model(self):
        """Access to the underlying model"""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def do_preprocess_traj(self):
        return False

    @property
    def has_jac(self):
        return False

    def density(self, x0: Union[float, np.ndarray], xt: Union[float, np.ndarray], t0: Union[float, np.ndarray], dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.exp(self._logdensity(x0, xt, t0, dt))

    @abstractmethod
    def _logdensity(self, x0: Union[float, np.ndarray], xt: Union[float, np.ndarray], t0: Union[float, np.ndarray], dt: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        The transition density evaluated at these arguments
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        raise NotImplementedError

    def __call__(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        return (-np.sum(np.maximum(self._min_prob, self._logdensity(x0=trj["x"][:-1], xt=trj["x"][1:], t0=0.0, dt=trj["dt"]))) / weight,)
