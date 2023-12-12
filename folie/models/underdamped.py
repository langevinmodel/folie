from typing import Optional
from abc import abstractmethod
import numpy as np

from ..base import Model


class ModelUnderdamped(Model):
    dim = 1

    def __init__(self, **kwargs):
        """
        Base model for overdamped Langevin equations, defined by

        dX(t) = mu(X,t)dt + sigma(X,t)dW_t

        :param has_exact_density: bool, set to true if an exact density is implemented
        """
        self._coefficients: Optional[np.ndarray] = None
        self.h = 1e-05

    @abstractmethod
    def force(self, x, t=0.0):
        """The force term of the model"""
        raise NotImplementedError

    @abstractmethod
    def friction(self, x, t=0.0):
        """The force term of the model"""
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x, t=0.0):
        """The diffusion term of the model"""
        raise NotImplementedError

    # ==============================
    # Direct acces to parameters (Not Implemented By Default)
    # ==============================

    @property
    def coefficients_force(self):
        """Access the coefficients"""
        return self._coefficients[: self._n_coeffs_force]

    @coefficients_force.setter
    def coefficients_force(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._coefficients[: self._n_coeffs_force] = vals.ravel()

    @property
    def coefficients_diffusion(self):
        """Access the coefficients"""
        return self._coefficients[self._n_coeffs_force :]

    @coefficients_diffusion.setter
    def coefficients_diffusion(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._coefficients[self._n_coeffs_force :] = vals.ravel()

    def force_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobien of the force with respect to coefficients
        """
        raise NotImplementedError

    def diffusion_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobien of the diffusion with respect to coefficients
        """
        raise NotImplementedError

    # ==============================
    # Derivatives (Numerical By Default)
    # ==============================

    def force_x(self, x, t: float = 0.0):
        """Calculate first spatial derivative of force, dmu/dx"""
        return (self.force(x + self.h, t) - self.force(x - self.h, t)) / (2 * self.h)

    def force_t(self, x, t: float = 0.0):
        """Calculate first time derivative of force, dmu/dt"""
        return (self.force(x, t + self.h) - self.force(x, t)) / self.h

    def force_xx(self, x, t: float = 0.0):
        """Calculate second spatial derivative of force, d^2mu/dx^2"""
        return (self.force(x + self.h, t) - 2 * self.force(x, t) + self.force(x - self.h, t)) / (self.h * self.h)

    def diffusion_x(self, x, t: float = 0.0):
        """Calculate first spatial derivative of diffusion term, dsigma/dx"""
        return (self.diffusion(x + self.h, t) - self.diffusion(x - self.h, t)) / (2 * self.h)

    def diffusion_xx(self, x, t: float = 0.0):
        """Calculate second spatial derivative of diffusion term, d^2sigma/dx^2"""
        return (self.diffusion(x + self.h, t) - 2 * self.diffusion(x, t) + self.diffusion(x - self.h, t)) / (self.h * self.h)
