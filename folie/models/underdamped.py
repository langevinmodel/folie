import numpy as np

from .overdamped import OverdampedFunctions


class UnderdampedFunctions(OverdampedFunctions):
    dim = 1

    def __init__(self, force, friction, diffusion, dim=1, **kwargs):
        """
        Base model for underdamped Langevin equations, defined by

        dX(t) = V(t)

        dV(t) = f(X,t)dt+ gamma(X,t)V(t)dt + sigma(X,t)dW_t

        """
        super().__init__(force, diffusion, dim=dim)
        self._friction = friction.reshape((self.dim, self.dim))
        self.coefficients = np.concatenate((np.zeros(self._n_coeffs_force), np.ones(self._n_coeffs_diffusion), np.ones(self._n_coeffs_friction)))

    def friction(self, x, t: float = 0.0):
        return self.friction(x[:, : self.dim_x])

    def friction_x(self, x, t: float = 0.0):
        return self.friction.grad_x(x[:, : self.dim_x])

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
