"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from .._numpy import np
from typing import Union
from .transitionDensity import TransitionDensity
import numba as nb


@nb.njit
def compute_va(trj, correct_jumps=False, jump=2 * np.pi, jump_thr=1.75 * np.pi, lamb_finite_diff=0.5, **kwargs):
    """
    Compute velocity by finite difference
    """

    diffs = trj["x"] - np.roll(trj["x"], 1, axis=0)
    dt = trj["dt"]
    if correct_jumps:
        diffs = np.where(diffs > -jump_thr, diffs, diffs + jump)
        diffs = np.where(diffs < jump_thr, diffs, diffs - jump)
        # raise NotImplementedError("Periodic data are not implemented yet")

    ddiffs = np.roll(diffs, -1, axis=0) - diffs
    sdiffs = lamb_finite_diff * np.roll(diffs, -1, axis=0) + (1.0 - lamb_finite_diff) * diffs

    trj["v"] = sdiffs["x"] / dt
    trj["a"] = ddiffs["x"] / dt**2
    return trj


class UnderdampedTransitionDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def preprocess_traj(self, trj, **kwargs):
        """
        Preprocess trajectories data
        """
        trj = compute_va(trj, **kwargs)
        if hasattr(self._model, "dim_h"):
            if self._model.dim_h > 0:
                trj["sig_h"] = np.zeros((trj["v"].shape[0], 2 * self._model.dim_h, 2 * self._model.dim_h))
                trj["v"] = np.concatenate((trj["v"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["a"] = np.concatenate((trj["a"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
        return trj


class BBKDensity(UnderdampedTransitionDensity):
    def __init__(self, model):
        """
        Class which represents the BBK approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def preprocess_traj(self, trj, **kwargs):
        """
        Preprocess trajectories data
        """
        trj = compute_va(trj, lamb_finite_diff=0.0, **kwargs)
        if hasattr(self._model, "dim_h"):
            if self._model.dim_h > 0:
                trj["sig_h"] = np.zeros((trj["v"].shape[0], 2 * self._model.dim_h, 2 * self._model.dim_h))
                trj["v"] = np.concatenate((trj["v"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["a"] = np.concatenate((trj["a"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
        return trj

    def _logdensity(self, x0, xt, v, a, dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        raise NotImplementedError


class VECDensity(UnderdampedTransitionDensity):
    def __init__(self, model):
        """
        Class which represents the VEC approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def preprocess_traj(self, trj, **kwargs):
        """
        Preprocess trajectories data
        """
        trj = compute_va(trj, **kwargs)
        trj["sig_h"] = np.zeros((trj["v"].shape[0], 2 * self._model.dim, 2 * self._model.dim))  # That would be dim_x+dim_h as the velocity is in the hidden dim
        if hasattr(self._model, "dim_h"):
            if self._model.dim_h > 0:
                trj["v"] = np.concatenate((trj["v"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["a"] = np.concatenate((trj["a"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
        return trj

    def _logdensity(self, x0, xt, v, a, dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Kessler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x0 and xt)
        """
        raise NotImplementedError
