from .overdamped import OverdampedFunctions
from .underdamped import UnderdampedFunctions
import numpy as np


class OverdampedHidden(OverdampedFunctions):
    """
    TODO: A class that implement an overdamped model with some extras hidden variables linearly correlated with visible ones


    d \\begin{pmatrix} X(t) \\ h(t) \\end{pmatrix} = f(X,t)dt+ gamma(X,t)h(t)dt + sigma(X,t)dW_t

    where

    f(X,t) is a $dim_x + dim_h$ vector

    gamma(X,t) is a $(dim_x + dim_h) \times dim_h$ matrix

    sigma(X,t) is a $(dim_x + dim_h) \times (dim_x + dim_h)$ matrix

    """

    def __init__(self, force, friction, diffusion, dim=1, dim_h=0, **kwargs):
        self.dim_h = dim_h
        self.dim_x = dim
        self._dim = self.dim_x + self.dim_h
        self._force = force.resize((self.dim,))
        if diffusion is None:
            self._diffusion = force.copy().resize((self.dim, self.dim))
        else:
            self._diffusion = diffusion.resize((self.dim, self.dim))
        self._friction = friction.resize((self.dim, self.dim_h))
        self._n_coeffs_force = self._force.size
        self._n_coeffs_diffusion = self._diffusion.size
        self._n_coeffs_friction = self._friction.size
        self.coefficients = np.concatenate((np.zeros(self._n_coeffs_force), np.ones(self._n_coeffs_friction), np.eye(self.dim).flatten()))

    def meandispl(self, x, t: float = 0.0):
        return self._force(x[:, : self.dim_x]) + np.einsum("tdh,th-> td", self.friction(x[:, : self.dim_x]), x[:, self.dim_x :])

    def meandispl_x(self, x, t: float = 0.0):
        return self._force.grad_x(x[:, : self.dim_x]) + np.einsum("tdhe,th-> tde", self._friction.grad_x(x[:, : self.dim_x]), x[:, self.dim_x :])

    def meandispl_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobian of the force with respect to coefficients
        """
        return np.concatenate((self._force.grad_coeffs(x[:, : self.dim_x]), np.einsum("tdhc,th-> tdc", self._friction.grad_coeffs(x[:, : self.dim_x]), x[:, self.dim_x :])), axis=-1)

    def force(self, x, t: float = 0.0):
        return self._force(x[:, : self.dim_x])

    def force_x(self, x, t: float = 0.0):
        return self._force.grad_x(x[:, : self.dim_x])

    def friction(self, x, t: float = 0.0):
        return self._friction(x[:, : self.dim_x])

    def friction_x(self, x, t: float = 0.0):
        return self._friction.grad_x(x[:, : self.dim_x])

    def friction_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobian of the force with respect to coefficients
        """
        return self._friction.grad_coeffs(x[:, : self.dim_x])

    def diffusion(self, x, t: float = 0.0):
        return self._diffusion(x[:, : self.dim_x])

    def diffusion_x(self, x, t: float = 0.0):
        return self._diffusion.grad_x(x[:, : self.dim_x])

    def diffusion_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobian of the diffusion with respect to coefficients
        """
        return self._diffusion.grad_coeffs(x[:, : self.dim_x])

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self._force.coefficients.ravel(), self._friction.coefficients.ravel(), self._diffusion.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._force.coefficients = vals.ravel()[: self._n_coeffs_force]
        self._friction.coefficients = vals.ravel()[self._n_coeffs_force : self._n_coeffs_force + self._n_coeffs_friction]
        self._diffusion.coefficients = vals.ravel()[self._n_coeffs_force + self._n_coeffs_friction :]

    @property
    def coefficients_friction(self):
        return self._friction.coefficients

    @coefficients_friction.setter
    def coefficients_friction(self, vals):
        self._friction.coefficients = vals


class UnderdampedHidden(UnderdampedFunctions):
    """
    TODO: A class that implement an underdamped model with some extras hidden variables linearly correlated with visible ones
    """

    def __init__(self, force, friction, diffusion, dim=1, dim_h=0, **kwargs):
        super().__init__(force, diffusion, dim=dim + dim_h)
        self.dim_h = dim_h
        self.dim_x = dim
