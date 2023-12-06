"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

import numpy as np
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

    ddiffs = np.shift(diffs, -1, axis=0) - diffs
    sdiffs = lamb_finite_diff * np.shift(diffs, -1, axis=0) + (1.0 - lamb_finite_diff) * diffs

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

    @property
    def do_preprocess_traj(self):
        return True

    def preprocess_traj(self, trj, **kwargs):
        """
        Preprocess trajectories data
        """
        return compute_va(trj, **kwargs)


class BBKDensity(UnderdampedTransitionDensity):
    def __init__(self, model):
        """
        Class which represents the BBK approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity(self, x0: Union[float, np.ndarray], xt: Union[float, np.ndarray], t0: Union[float, np.ndarray], dt: float) -> Union[float, np.ndarray]:
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

    def _logdensity(self, x0: Union[float, np.ndarray], xt: Union[float, np.ndarray], t0: Union[float, np.ndarray], dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Kessler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x0 and xt)
        """
        raise NotImplementedError
