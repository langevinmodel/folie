from typing import Optional
from abc import abstractmethod
import numpy as np
from scipy.stats import norm

from ..base import Model

# TODO: Implement multidimensionnal version


class ModelOverdamped(Model):
    def __init__(self, has_exact_density: bool = False, **kwargs):
        """
        Base model for overdamped Langevin equations, defined by

        dX(t) = mu(X,t)dt + sigma(X,t)dW_t

        :param has_exact_density: bool, set to true if an exact density is implemented
        """
        self._has_exact_density = has_exact_density
        self._coefficients: Optional[np.ndarray] = None
        self.h = 1e-05

    @abstractmethod
    def force(self, x, t=0.0):
        """The force term of the model"""
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, x, t=0.0):
        """The diffusion term of the model"""
        raise NotImplementedError

    # ==============================
    # Exact Transition Density and Simulation Step, override when available
    # ==============================

    @property
    def has_exact_density(self) -> bool:
        """Return true if model has an exact density implemented"""
        return self._has_exact_density

    def exact_density(self, x0: float, xt: float, t0: float, dt: float = 0.0) -> float:
        """
        In the case where the exact transition density,
        P(Xt, t | X0) is known, override this method
        :param x0: float, the current value
        :param xt: float, the value to transition to
        :param t0: float, the time of observing x0
        :param dt: float, the time step between x0 and xt
        :return: probability
        """
        raise NotImplementedError

    def exact_step(self, x, dt, dZ, t=0.0):
        """Exact Simulation Step, Implement if known (e.g. Browian motion or GBM)"""
        raise NotImplementedError

    # ==============================
    # Direct acces to parameters (Not Imlemented By Default)
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


class BrownianMotion(ModelOverdamped):
    """
    Model for (forceed) Brownian Motion
    Parameters:  [mu, sigma]

    dX(t) = mu(X,t)dt + sigma(X,t)dW_t

    where:
        mu(X,t)    = mu   (constant)
        sigma(X,t) = sqrt(sigma)   (constant, >0)
    """

    _n_coeffs_force = 1

    def __init__(self, **kwargs):
        super().__init__(has_exact_density=True, default_sim_method="Exact")
        self.coefficients = np.array([0.0, 1.0])

    def force(self, x, t: float = 0.0):
        return self._coefficients[0] * (x > -10000)  # todo: reshape?

    def diffusion(self, x, t: float = 0.0):
        return self._coefficients[1] * (x > -10000)

    def exact_density(self, x0: float, xt: float, t0: float, dt: float = 0.0) -> float:
        mu, sigma2 = self._coefficients
        mean_ = x0 + mu * dt
        return norm.pdf(xt, loc=mean_, scale=np.sqrt(sigma2 * dt))

    def exact_step(self, x, dt, dZ, t=0.0):
        """Simple Brownian motion can be simulated exactly"""
        sig_sq_dt = np.sqrt(self._coefficients[1] * dt)
        return x + self._coefficients[0] * dt + sig_sq_dt * dZ

    def is_linear(self):
        return True

    def force_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobien of the force with respect to coefficients
        """
        raise np.ones_like(x)

    def diffusion_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobien of the diffusion with respect to coefficients
        """
        raise np.ones_like(x)

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def force_t(self, x, t: float = 0.0):
        return 0.0

    def diffusion_x(self, x, t: float = 0.0):
        return np.zeros_like(x)

    def diffusion_xx(self, x, t: float = 0.0):
        return np.zeros_like(x)


class OrnsteinUhlenbeck(ModelOverdamped):
    """
    Model for OU (ornstein-uhlenbeck):
    Parameters: [kappa, mu, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = kappa * (theta - X)
        sigma(X,t) = sqrt(sigma)
    """

    _n_coeffs_force = 2

    def __init__(self, **kwargs):
        super().__init__(has_exact_density=True)
        self.coefficients = np.array([1.0, 0.0, 1.0])

    def force(self, x, t: float = 0.0):
        return self._coefficients[0] * (self._coefficients[1] - x)

    def diffusion(self, x, t: float = 0.0):
        return self._coefficients[2]

    def exact_density(self, x0: float, xt: float, t0: float, dt: float = 0.0) -> float:
        kappa, theta, sigma = self._coefficients
        mu = theta + (x0 - theta) * np.exp(-kappa * dt)
        # mu = X0*np.exp(-kappa*t) + theta*(1 - np.exp(-kappa*t))
        var = (1 - np.exp(-2 * kappa * dt)) * (sigma / (2 * kappa))
        return norm.pdf(xt, loc=mu, scale=np.sqrt(var))

    def exact_step(self, x, dt, dZ, t=0.0):
        kappa, theta, sigma = self._coefficients
        mu = theta + (x - theta) * np.exp(-kappa * dt)
        var = (1 - np.exp(-2 * kappa * dt)) * (sigma / (2 * kappa))
        return mu * dt + np.sqrt(var * dt) * dZ

    # =======================
    # (Optional) Overrides for numerical derivatives to improve performance
    # =======================

    def force_t(self, x, t: float = 0.0):
        return 0.0

    def diffusion_x(self, x, t: float = 0.0):
        return np.zeros_like(x)

    def diffusion_xx(self, x, t: float = 0.0):
        return np.zeros_like(x)


class OverdampedBF(ModelOverdamped):
    """
    A class that implement a overdamped model with basis function
    """

    def __init__(self, basis, **kwargs):
        super().__init__()
        self.basis = basis
        self._n_coeffs_force = self.basis.n_output_features_
        self.coefficients = np.concatenate((np.zeros(self._n_coeffs_force), np.ones(self._n_coeffs_force)))

    def force(self, x, t: float = 0.0):
        return np.dot(self.basis(x), self.coefficients_force).reshape(-1, 1)

    def diffusion(self, x, t: float = 0.0):
        return np.dot(self.basis(x), self.coefficients_diffusion).reshape(-1, 1)

    def force_t(self, x, t: float = 0.0):
        return 0.0

    def force_x(self, x, t: float = 0.0):
        return np.dot(self.coefficients_force, self.basis.deriv(x))

    def force_xx(self, x, t: float = 0.0):
        return np.dot(self.coefficients_force, self.basis.hessian(x))

    def diffusion_t(self, x, t: float = 0.0):
        return 0.0

    def diffusion_x(self, x, t: float = 0.0):
        return np.dot(self.coefficients_diffusion, self.basis.deriv(x))

    def diffusion_xx(self, x, t: float = 0.0):
        return np.dot(self.coefficients_diffusion, self.basis.hessian(x))

    def is_linear(self):
        return True

    def force_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobien of the force with respect to coefficients
        """
        return self.basis(x)

    def diffusion_jac_coeffs(self, x, t: float = 0.0):
        """
        Jacobien of the diffusion with respect to coefficients
        """
        return self.basis(x)
