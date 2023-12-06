import numpy as np
import numba as nb

from .overdamped_transitionDensity import EulerDensity
from ..models.piecewise_overdamped import linear_interpolation_with_gradient


class EulerNumbaOptimizedDensity(EulerDensity):
    def __init__(self, model):
        """
        Class which represents the exact transition density for a model (when available)
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    @property
    def do_preprocess_traj(self):
        return True

    @property
    def has_jac(self):
        return True

    def preprocess_traj(self, trj, **kwargs):
        """
        Preprocess trajectories data
        """
        trj["pre"] = self.model.preprocess_traj(trj["x"], **kwargs)
        return trj

    def __call__(self, weight, trj, coefficients):
        """
        The exact transition density (when applicable)
        Note: this will raise exception if the model does not implement exact_density
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        return objective_order1_debiased(coefficients, self.model.knots, trj["pre"], trj["dt"], self.model.beta, None) / weight


@nb.njit(parallel=True)
def objective_order1_debiased(coefficients, knots, traj, dt, beta, f):
    """Objective function: order-1 OptLE for overdamped Langevin, order-1 propagator
    Includes the debiasing feature of Hallegot, Pietrucci and Hénin for time-dependent biases

    Args:
        coefficients (ndarray): parameters of the model - piecewise-linear grad F (free energy) and log D
        knots (ndarray): CV values forming the knots of the piecewise-linear approximation of logD and gradF
        q (list of ndarray): trajectories of the CV
        deltaq (list of ndarray): trajectories of CV differences
        f (list of ndarray): trajectories of the biasing force

    Returns:
        real, ndarray: objective function and its derivatives with respect to model parameters
    """

    idx, h, deltaq = traj
    G, logD, dXdk = linear_interpolation_with_gradient(idx, h, knots, coefficients)
    # dXdk is the gradient with respect to the knots (same for all quantities)

    # Debiasing (truncate last traj point)
    # G -= f[i][:-1]

    phi = -beta * np.exp(logD) * G * dt
    dphidlD = -beta * np.exp(logD) * G * dt
    dphidG = -beta * np.exp(logD) * dt

    mu = 2.0 * np.exp(logD) * dt
    dmudlD = 2.0 * np.exp(logD) * dt
    logL = (0.5 * logD + np.square(deltaq - phi) / (2.0 * mu)).sum()
    dLdlD = 0.5 + (2.0 * (deltaq - phi) * -1.0 * dphidlD * (2.0 * mu) - np.square(deltaq - phi) * 2.0 * dmudlD) / np.square(2.0 * mu)
    dLdG = 2.0 * (deltaq - phi) * -1.0 * dphidG / (2.0 * mu)

    dlogLdkG = np.dot(dXdk, dLdG)
    dlogLdklD = np.dot(dXdk, dLdlD)

    return logL, np.concatenate((dlogLdkG, dlogLdklD))
