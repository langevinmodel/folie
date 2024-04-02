"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from .._numpy import np
import warnings
from .transitionDensity import TransitionDensity

from torch.nn import functional as F


class EulerDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def forward(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        like, jac = self._logdensity(**trj)
        x = trj["x"]
        xt = trj["xt"]
        bias = trj["bias"]
        sig2t = (self._model.diffusion(x)) * self.dt
        mut = self._model.meandispl(x, bias) * self.dt  # Faut trouver comme jt rentre les données dans ce cas
        return F.gaussian_nll_loss(xt, mut, sig2t, eps=self._min_prob)  # Ca fait déjà la somme donc pas besoin de la refaire

    def _hiddenvariance(self, x, xt, sigh, dt, **kwargs):
        """
        The transition density evaluated at these arguments
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """

        E2 = self._model.friction(x[:, : self._model.dim_x], **kwargs) * dt
        V = (self._model.diffusion(x[:, : self._model.dim_x], **kwargs)) * dt
        invV = np.linalg.inv(V)
        dh = self._model.dim_h
        dhdh = sigh[:, :dh, :dh] - sigh[:, :dh, dh:] - sigh[:, dh:, :dh] + sigh[:, dh:, dh:]
        hdh = sigh[:, dh:, :dh] - sigh[:, dh:, dh:]
        dhh = sigh[:, :dh, dh:] - sigh[:, dh:, dh:]
        hh = sigh[:, dh:, dh:]

        EVE = np.einsum("tdh,tdf,tfg-> thg", E2, invV, E2)
        EV = np.einsum("tdh,tdf-> thf", E2, invV)
        VE = np.einsum("tdf,tfh-> tdh", invV, E2)

        extra_ll = -0.5 * np.einsum("tij,tji->t", invV[:, dh:, dh:], dhdh) + 0.5 * np.einsum("tij,tji->t", EV[:, :, dh:], dhh) + 0.5 * np.einsum("tij,tji->t", VE[:, dh:, :], hdh) - 0.5 * np.einsum("tij,tji->t", EVE, hh)
        jacE2 = self._model.friction.grad_coeffs(x[:, : self._model.dim_x], **kwargs) * dt

        EVjE = np.einsum("tdh,tdf,tfgc-> thgc", E2, invV, jacE2)
        jEV = np.einsum("tdhc,tdf-> thfc", jacE2, invV)
        VjE = np.einsum("tdf,tfhc-> tdhc", invV, jacE2)

        l_jac_E = -0.5 * np.einsum("tijc,tji->tc", jEV[:, :, dh:, :], dhh) - 0.5 * np.einsum("tijc,tji->tc", VjE[:, dh:, ...], hdh) + np.einsum("tijc,tji->tc", EVjE, hh)

        jacV = self._model.diffusion.grad_coeffs(x[:, : self._model.dim_x], **kwargs) * dt
        jacinvV = -np.einsum("tij,tjkc,tkl->tilc", invV, jacV, invV)
        EjVE = np.einsum("tdh,tdfc,tfg-> thgc", E2, jacinvV, E2)
        EjV = np.einsum("tdh,tdfc-> thfc", E2, jacinvV)
        jVE = np.einsum("tdfc,tfh-> tdhc", jacinvV, E2)
        l_jac_V = 0.5 * np.einsum("tijc,tji->tc", jacinvV[:, dh:, dh:, :], dhdh) - 0.5 * np.einsum("tijc,tji->tc", jVE[:, dh:, :], hdh) - 0.5 * np.einsum("tijc,tji->tc", EjV[:, :, dh:], dhh) + 0.5 * np.einsum("tijc,tji->tc", EjVE, hh)
        return extra_ll, np.concatenate((l_jac_E, l_jac_V), axis=-1)

    def hiddencorrection(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        like, jac = self._hiddenvariance(x=trj["x"], xt=trj["xt"], sigh=trj["sig_h"], dt=trj["dt"])
        return like.sum() / weight, -np.hstack((np.zeros(self._model.force.size), jac.sum(axis=0) / weight))

    def e_step(self, weight, trj, coefficients, mu0, sig0):
        """
        In presence of hidden variables, reconstruct then using a Kalman Filter.
        Assume that the model is an OverdampedHidden model
        """
        self._model.coefficients = coefficients
        muh, Sigh = filtersmoother(
            trj["xt"][:, : self._model.dim_x],
            self._model.force(trj["x"][:, : self._model.dim_x], trj["bias"][:, : self._model.dim_x]) * trj["dt"],
            self._model.friction(trj["x"][:, : self._model.dim_x]) * trj["dt"],
            self._model.diffusion(trj["x"][:, : self._model.dim_x]) * trj["dt"],
            mu0,
            sig0,
        )

        trj["sig_h"] = Sigh
        trj["x"][:, self._model.dim_x :] = muh[:, self._model.dim_h :]
        trj["xt"][:, self._model.dim_x :] = muh[:, : self._model.dim_h]
        return muh[0, self._model.dim_h :] / weight, Sigh[0, self._model.dim_h :, self._model.dim_h :] / weight  # Return µ0 and sig0
