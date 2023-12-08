from .overdamped import ModelOverdamped
import numpy as np


class OverdampedHidden(ModelOverdamped):
    """
    TODO: A class that implement an overdamped model with some extras hidden variables linearly correlated with visible ones
    """

    def __init__(self, model, dim_h, **kwargs):
        super().__init__()
        self.inner_model = model
        self.dim_h = dim_h
        self.dim_x = self.inner_model.dim
        self.dim = self.dim_x + self.dim_h
        self._n_coeffs_force = self.inner_model._n_coeffs_force + (self.dim_h + 1)
        self.coefficients = np.concatenate((np.zeros(self._n_coeffs_force), np.ones(self._n_coeffs_force)))

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._coefficients = vals

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

    def force(self, x, t: float = 0.0):
        f_inner = self.inner_model.force(x[:, : self.dim_x], t)
        G, logD, _ = linear_interpolation_with_gradient(idx, h, self.knots, self._coefficients)
        return -self.beta * np.exp(logD) * G

    def diffusion(self, x, t: float = 0.0):
        self.inner_model.diffusion(x, t)
        G, logD, _ = linear_interpolation_with_gradient(idx, h, self.knots, self._coefficients)
        return 2.0 * np.exp(logD)

    def force_t(self, x, t: float = 0.0):
        return 0.0

    def force_x(self, x, t: float = 0.0):
        self.inner_model.force_x(x, t)
        return np.dot(self._coefficients[: self._n_coeffs_force], self.basis.derivative(x))

    def force_xx(self, x, t: float = 0.0):
        return 0.0

    def diffusion_t(self, x, t: float = 0.0):
        return 0.0

    def diffusion_x(self, x, t: float = 0.0):
        idx, h, _ = self.preprocess_traj(x)
        G, logD, dXdk = linear_interpolation_with_gradient(idx, h, self.knots, self._coefficients)
        return np.dot(self._coefficients[self._n_coeffs_force :], self.basis.derivative(x))

    def diffusion_xx(self, x, t: float = 0.0):
        return 0.0

    def is_linear(self):
        return True
