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
        super().__init__(force, diffusion, dim=dim + dim_h)
        self.dim_h = dim_h
        self.dim_x = dim
        self.dim = self.dim_x + self.dim_h
        self._friction = friction.reshape((self.dim, self.dim_h), force_reshape=True)
        self._n_coeffs_friction = self._friction.size
        self.coefficients = np.concatenate((np.zeros(self._n_coeffs_force), np.ones(self._n_coeffs_diffusion), np.ones(self._n_coeffs_friction)))

    def force(self, x, t: float = 0.0):
        return self._force(x[:, : self.dim_x]) + np.einsum("tdh,th-> td", self.friction(x[:, : self.dim_x]), x[:, self.dim_x :])

    def force_x(self, x, t: float = 0.0):
        return self._force.grad_x(x[:, : self.dim_x]) + np.einsum("tdhe,th-> tde", self.friction.grad_x(x[:, : self.dim_x]), x[:, self.dim_x :])

    def force_visible_part(self, x, t: float = 0.0):
        return self._force(x[:, : self.dim_x])

    def force_visible_part_x(self, x, t: float = 0.0):
        return self._force.grad_x(x[:, : self.dim_x])

    def friction(self, x, t: float = 0.0):
        return self._friction(x[:, : self.dim_x])

    def friction_x(self, x, t: float = 0.0):
        return self._friction.grad_x(x[:, : self.dim_x])

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self._force.coefficients.ravel(), self._diffusion.coefficients.ravel(), self._friction.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._force.coefficients = vals.ravel()[: self._n_coeffs_force]
        self._diffusion.coefficients = vals.ravel()[self._n_coeffs_force : self._n_coeffs_force + self._n_coeffs_diffusion]
        self._friction.coefficients = vals.ravel()[self._n_coeffs_force + self._n_coeffs_diffusion :]

    @property
    def coefficients_friction(self):
        return self._force.coefficients

    @coefficients_friction.setter
    def coefficients_friction(self, vals):
        self._friction.coefficients = vals

    def is_linear(self):
        return self._force.is_linear and self.friction.is_linear and self.diffusion.is_linear


class UnderdampedHidden(UnderdampedFunctions):
    """
    TODO: A class that implement an underdamped model with some extras hidden variables linearly correlated with visible ones
    """

    def __init__(self, force, friction, diffusion, dim=1, dim_h=0, **kwargs):
        super().__init__(force, diffusion, dim=dim + dim_h)
        self.dim_h = dim_h
        self.dim_x = dim
