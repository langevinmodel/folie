"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from .._numpy import np
from typing import Union
from .transitionDensity import TransitionDensity
from .transitionDensity import gaussian_likelihood_1D, gaussian_likelihood_ND, gaussian_likelihood_derivative_1D, gaussian_likelihood_derivative_ND
import numba as nb


@nb.njit
def compute_va(trj, correct_jumps=False, jump=2 * np.pi, jump_thr=1.75 * np.pi, lamb_finite_diff=0.5, **kwargs):
    """
    Compute velocity by finite differences if an exact velocity is not available.
    """
    # If v is already in the data, don't replace it by finite diff.
    if "v" not in trj:
        diffs = trj["x"] - np.roll(trj["x"], 1, axis=0)
        dt = trj["dt"]
        if correct_jumps:
            diffs = np.where(diffs > -jump_thr, diffs, diffs + jump)
            diffs = np.where(diffs < jump_thr, diffs, diffs - jump)
        # raise NotImplementedError("Periodic data are not implemented yet")

        ddiffs = np.roll(diffs, -1, axis=0) - diffs
        sdiffs = lamb_finite_diff * np.roll(diffs, -1, axis=0) + (1.0 - lamb_finite_diff) * diffs
        # Former version : sdiffs and ddiffs are np arrays, not dicts
        #trj["v"] = sdiffs["x"] / dt   
        #trj["a"] = ddiffs["x"] / dt**2  
        trj["v"] = sdiffs / dt
        trj["a"] = ddiffs / dt**2
    elif "a" not in trj:
        dv = trj["v"] - np.roll(trj["v"],1,axis=0)
        dt = trj["dt"]
        vdiffs = lamb_finite_diff * np.roll(dv, -1, axis=0) + (1.0 - lamb_finite_diff) * dv
        
        trj["a"] = vdiffs / dt

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

    def _logdensity(self, x, xt, v, a, dt: float) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Euler expansion
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """
        raise NotImplementedError


class VECDensity(UnderdampedTransitionDensity):
    use_jac = False
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
        #trj = compute_va(trj, **kwargs)
        if "xt" not in trj:
            trj["xt"] = trj["x"][2:-1]
            trj["x"] = trj["x"][1:-2]
            # when v is computed by finite difference, v[0] and v[-1] have no meaning and should be discarded ; which means we should do the same for x (so that trajectories on x and v are compatible in shape).
        if "vt" not in trj:
            trj["vt"] = trj["v"][2:-1]
            trj["v"] = trj["v"][1:-2]
        
        if "a" not in trj:
            trj["a"] = trj["a"][1:-2]

        if "bias" not in trj:
            trj["bias"] = np.zeros((1, trj["x"].shape[1]))

        trj["sig_h"] = np.zeros((trj["v"].shape[0], 2 * self._model.dim, 2 * self._model.dim))  # That would be dim_x+dim_h as the velocity is in the hidden dim
        if hasattr(self._model, "dim_h"):
            if self._model.dim_h > 0:
                trj["v"] = np.concatenate((trj["v"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["a"] = np.concatenate((trj["a"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["vt"] = np.concatenate((trj["vt"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["x"] = np.concatenate((trj["x"], np.zeros((trj["x"].shape[0], self._model.dim_h))), axis=1)
                trj["xt"] = np.concatenate((trj["xt"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["bias"] = np.concatenate((trj["bias"], np.zeros((trj["bias"].shape[0], self._model.dim_h))), axis=1)
        return trj

# changed a to vt bcs why is a in this and not vt (does not seem consistent with the propagator)
    def _logdensity1D(self, x, xt, v, vt, dt: float, bias=0.0, **kwargs) -> Union[float, np.ndarray]:
        """
        The transition density obtained via Kessler expansion
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x and xt)
        """
        # Drift and derivatives
        mu = self._model.drift(x,v).ravel()                #  b
        mu_x = self._model.drift.grad_x(x,v).ravel()       # b_q
        mu_xx = self._model.drift.hessian_x(x,v).ravel()   # b_qq
        
        # Friction and derivatives
        gamma = self._model.friction(x,v).ravel()          # b_v
        gamma_x = self._model.friction.grad_x(x,v).ravel() # b_qv

        # Diffusion and derivatives
        c = self._model.diffusion(x,v).ravel()             # c
        c_x = self._model.diffusion.grad_x(x,v).ravel()    # c_q
        c_xx = self._model.diffusion.hessian_x(x,v).ravel()# c_qq

        X = np.einsum("i...-> ...i", [x,v])                # X = [q,v].T
        Xt = np.einsum("i...-> ...i", [xt,vt])             # Xt = [qt, vt].T

        # First-order cumulants (from Girardier et al., JCP 2023)
        # <q>
        x_exp = x + v * dt + mu * dt**2 / 2 + (mu_x * v + mu * gamma) * dt**3 / 6
        # <v>
        v_exp = v + mu * dt + (mu_x * v + mu * gamma) * dt**2 / 2 + (mu_xx * v**2 + mu_x * gamma * v + 2 * mu * gamma_x * v + mu * mu_x + mu * gamma**2 + 2 * c * gamma_x) * dt**3 / 6
        E = np.einsum("i...-> ...i", [x_exp, v_exp])

        # Second-order cumulants
        Mxx = 2 * c * dt**3
        Mvv = 2 * c * dt + (c_x * v + 2 * c * gamma) * dt**2 + (c_xx * v**2 + 2 * c_x * gamma * v + 4 * c * gamma_x * v + c_x * mu + 2 * c * mu_x + 4 * c * gamma**2) * dt**3 / 3
        Mxv = c * dt**2 + (c_x * v + 3 * c * gamma) * dt**3 / 3
        print("Mxx : {Mxx.shape}")
        print("Mxv : {Mxv.shape}")
        print("Mvv : {Mvv.shape}")
        M = np.einsum("ij...-> ...ij", [[Mxx, Mxv], [Mxv, Mvv]])  # Diffusion matrix

        if not use_jac:
            return gaussian_likelihood_ND(Xt, E, M)
        else:
            # Jacobians of all quantities above
            # Drift and derivatives
            jac_mu = self._model.drift.grad_coeffs(x,v).ravel()                #  b
            jac_mu_x = self._model.drift.grad_x.grad_coeffs(x,v).ravel()       # b_q
            jac_mu_xx = self._model.drift.hessian_x.grad_coeffs(x,v).ravel()   # b_qq
            _
            # Friction and derivatives
            jac_gamma = self._model.friction.grad_coeffs(x,v).ravel()          # b_v
            jac_gamma_x = self._model.friction.grad_x.grad_coeffs(x,v).ravel() # b_qv
    
            # Diffusion and derivatives
            jac_c = self._model.diffusion.grad_coeffs(x,v).ravel()             # c
            jac_c_x = self._model.diffusion.grad_x.grad_coeffs(x,v).ravel()    # c_q
            jac_c_xx = self._model.diffusion.hessian_x.grad_coeffs(x,v).ravel()# c_qq

            # First-order cumulants
            jac_x_exp = jac_mu * dt**2 / 2 + (v * jac_mu_x + gamma * jac_mu + mu * jac_gamma) * dt**3 / 6
            jac_v_exp = jac_mu * dt + (v * jac_mu_x + gamma * jac_mu + mu * jac_gamma) * dt**2 / 2 + (v**2*jac_mu_xx + v * (gamma * jac_mu_x + mu_x * jac_gamma) + 2 * v * (gamma_x * jac_mu + mu * jac_gamma_x ) + mu * jac_mu_x + jac_mu * mu_x + gamma**2 * jac_mu + 2 * mu * gamma * jac_gamma + 2 * c * jac_gamma_x + 2 * gamma_x * jac_c) * dt**3 / 6
            jacE = np.einsum("i...-> ...i", [jac_x_exp, jac_v_exp])

            # Second-order cumulants
            jac_Mxx = jac_c * dt**3 * 2 / 3
            jac_Mvv = 2 * jac_c * dt + (v*jac_c_x + 2 * gamma * jac_c + 2 * c * jac_gamma) * dt**2 + (v**2*jac_c_xx + 2 * v * c_x * jac_gamma + 2 * v * jac_c_x * gamma + 4 * jac_c * gamma_x * v + 4 * c * jac_gamma_x * v + c_x * jac_mu + mu * jac_c_x + 2 * jac_c * mu_x + 2 * jac_mu_x * c + 4 * gamma**2 * jac_c + 8 * gamma * jac_gamma * c) * dt**3 / 3
            jac_Mxv = jac_c * dt**2 + (jac_c_x * v + 3 * jac_c * gamma + 3 * jac_gamma * c) * dt**3 / 3
            jacM = np.einsum("ij...-> ...ij", [[jac_Mxx, jac_Mxv], [jac_Mxv, jac_Mvv]])  # Diffusion matrix

        # Here we kind of divert likelihood_ND from its original scope
        # (treating multi-dimensional CVs) by defining X = (x,v) a 2D-
        # variable encapsulating position and velocity to yield a more
        # compact formulation of the likelihood (i.e., a gaussian).
        # As such, likelihood_ND is not adapted for multi-dimensional
        # underdamped models. But maybe with some minimal changes (re-
        # -placing explicit indices by ellipses in np.einsum, e.g.
        # "tij,tj-> ti" should become "t...j,t...j-> t...") we could 
        # make it work.
            return gaussian_likelihood_ND_derivatives(Xt, E, M, jacE, jacM)

        
        #raise NotImplementedError
