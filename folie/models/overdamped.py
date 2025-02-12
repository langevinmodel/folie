from .._numpy import np
import warnings
from scipy.stats import norm

from ..base import Model
from ..functions import Constant, Polynomial, BSplinesFunction, ParametricFunction, ModelOverlay
from ..domains import Domain


class BaseModelOverdamped(Model):
    _has_exact_density = False

    def __init__(self, dim=1, has_bias=False, **kwargs):
        r"""
        Base model for overdamped Langevin equations.

        The evolution equation for variable X(t) is defined as

        .. math::

            \mathrm{d}X(t) = F(X)\mathrm{d}t + sigma(X,t)\mathrm{d}W_t

        The components of the overdamped model are the drift profile F(X) as well as the diffusion :math: `D(x) = \sigma(X)\sigma(X)^\T`

        When considering equilibrium model, the drift and diffusion profile are related to the free energy profile V(X) via

        .. math::
            F(x) = -D(x) \nabla V(x) + \mathrm{div} D(x)

        """

        self._dim = dim
        self.is_biased = False

        if self.dim <= 1:
            output_shape_drift = ()
            output_shape_diff = ()
        else:
            output_shape_drift = (self.dim,)
            output_shape_diff = (self.dim, self.dim)

        if hasattr(self, "_drift") and hasattr(self, "_diffusion"):
            self.pos_drift = ModelOverlay(self, "_pos_drift", output_shape=output_shape_drift)
            self.diffusion = ModelOverlay(self, "_diffusion", output_shape=output_shape_diff)

        self.drift = ModelOverlay(self, "_drift", output_shape=output_shape_drift)

        if has_bias:
            self.add_bias(has_bias)

    # ==============================
    # Exact Transition Density and Simulation Step, override when available
    # ==============================

    @property
    def has_exact_density(self) -> bool:
        """Return true if model has an exact density implemented"""
        return self._has_exact_density

    @property
    def dim(self):
        """
        Dimensionnality of the model
        """
        return self._dim

    @dim.setter
    def dim(self, dim):
        """
        Dimensionnality of the model
        """
        if dim == 0:
            dim = 1
        if dim != self._dim:
            raise ValueError("Dimension did not match dimension of the model. Change model or review dimension of your data")

    def preprocess_traj(self, trj, **kwargs):
        return trj

    def add_bias(self, bias=True):
        if self.dim <= 1:
            output_shape_drift = ()
        else:
            output_shape_drift = (self.dim,)
        self.drift = ModelOverlay(self, "_drift_biased", output_shape=output_shape_drift)
        self.is_biased = True

    def remove_bias(self):
        if self.is_biased:
            if self.dim <= 1:
                output_shape_drift = ()
            else:
                output_shape_drift = (self.dim,)
            self.drift = ModelOverlay(self, "_drift", output_shape=output_shape_drift)
            self.is_biased = False

    def _drift(self, x, *args, **kwargs):
        return self.pos_drift(x, *args, **kwargs)

    def _drift_dx(self, x, *args, **kwargs):
        return self.pos_drift.grad_x(x, *args, **kwargs)

    def _drift_d2x(self, x, *args, **kwargs):
        return self.pos_drift.hessian_x(x, *args, **kwargs)

    def _drift_dcoeffs(self, x, *args, **kwargs):
        """
        Jacobian of the drift with respect to coefficients
        """
        return self.pos_drift.grad_coeffs(x, *args, **kwargs)

    @property
    def coefficients_drift(self):
        """Access the coefficients"""
        return self.pos_drift.coefficients

    @coefficients_drift.setter
    def coefficients_drift(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals

    def _drift_biased(self, x, bias, *args, **kwargs):
        fx = self.pos_drift(x, *args, **kwargs)
        return fx + np.einsum("t...h,th-> t...", self.diffusion(x, *args, **kwargs).reshape((*fx.shape, bias.shape[1])), bias)

    def _drift_biased_dx(self, x, bias, *args, **kwargs):
        dfx = self.pos_drift.grad_x(x, *args, **kwargs)
        return dfx + np.einsum("t...he,th-> t...e", self.diffusion.grad_x(x, *args, **kwargs).reshape((*dfx.shape[:-1], bias.shape[1], dfx.shape[-1])), bias)

    def _drift_biased_d2x(self, x, bias, *args, **kwargs):
        ddfx = self.pos_drift.hessian_x(x, *args, **kwargs)
        return ddfx + np.einsum("t...hef,th-> t...ef", self.diffusion.hessian_x(x, *args, **kwargs).reshape((*ddfx.shape[:-2], bias.shape[1], *ddfx.shape[-2:])), bias)

    def _drift_biased_dcoeffs(self, x, bias, *args, **kwargs):
        """
        Jacobian of the drift with respect to coefficients
        """
        return self.pos_drift.grad_coeffs(x, *args, **kwargs)

    @property
    def coefficients_drift_biased(self):
        """Access the coefficients"""
        return self.pos_drift.coefficients

    @coefficients_drift_biased.setter
    def coefficients_drift_biased(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.drift.coefficients.ravel(), self.diffusion.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.drift.coefficients = vals.ravel()[: self.pos_drift.size]
        self.diffusion.coefficients = vals.ravel()[self.pos_drift.size : self.pos_drift.size + self.diffusion.size]


class Overdamped(BaseModelOverdamped):
    r"""
    A class that implement a overdamped model with given functions for space dependency


    The evolution equation for variable X(t) is defined as

    .. math::

        \mathrm{d}X(t) = F(X)\mathrm{d}t + \sigma(X)\mathrm{d}W_t

    The components of the overdamped model are the drift profile F(X) as well as the diffusion :math:`D(x) =  \frac{1}{2} \sigma(X)\sigma(X)^T`

    When considering equilibrium models, the drift and diffusion profile are related to the free energy profile V(X) via

    .. math::
        F(x) = -D(x) \nabla V(x) + \mathrm{div} D(x)

    """

    def __init__(self, drift, diffusion=None, dim=None, **kwargs):
        r"""
        Parameters
        ----------
        drift, diffusion : Functions
            Functions for the spatial dependance of the drift :math:`F(x)` and diffusion :math:`D(x)`.
            If diffusion is not given it default to the copy of drift

        dim : int
            Dimension of the model. By default it is the dimension of the domain of the drift
        has_bias: None, bool
            If None, assume no bias in the data.
            If true, this assume that an extra column is present in the data

        """
        if dim is None:
            dim = drift.domain.dim
        super().__init__(dim=dim, **kwargs)
        if dim > 1:
            drift_shape = (dim,)
            diffusion_shape = (dim, dim)
        else:
            drift_shape = ()
            diffusion_shape = ()
        self.pos_drift = drift.resize(drift_shape)
        if diffusion is None or diffusion is drift:
            self.diffusion = drift.copy().resize(diffusion_shape)
        else:
            self.diffusion = diffusion.resize(diffusion_shape)

    @BaseModelOverdamped.dim.setter
    def dim(self, dim):
        if dim > 1:
            drift_shape = (dim,)
            diffusion_shape = (dim, dim)
        else:
            drift_shape = ()
            diffusion_shape = ()
        self.pos_drift = self.pos_drift.resize(drift_shape)
        self.diffusion = self.diffusion.resize(diffusion_shape)
        self._dim = dim

    def preprocess_traj(self, trj, **kwargs):
        if hasattr(self.pos_drift.domain, "localize_data") or hasattr(self.diffusion.domain, "localize_data"):
            # Check if domain are compatible
            cells_idx, loc_x = self.pos_drift.domain.localize_data(trj["x"], **kwargs)
            trj["cells_idx"] = cells_idx
            trj["loc_x"] = loc_x
        return trj


#  Set of quick interface to more common models


class BrownianMotion(Overdamped):
    r"""
    Model for (forced) Brownian Motion
    Parameters:  :math:`[\mu, \sigma]`

    .. math::
        dX(t) = \mu(X)dt + \sigma(X)dW_t

    where:
        :math:`\mu(X)    = \mu`

        :math:`\sigma(X) = \sqrt{2\sigma}`
    """

    _has_exact_density = True

    def __init__(self, mu=0, sigma=1.0, dim=1, **kwargs):
        """
        Parameters
        ----------

            mu: float or ndarray of shape (dim,)

            sigma: float or ndarray of shape (dim,dim)
                constant, >0

            dim: int
                Wanted dimension of the model

        """
        super().__init__(Constant(domain=Domain.Rd(dim)), Constant(domain=Domain.Rd(dim)), dim=dim, **kwargs)
        self.pos_drift.coefficients = mu * np.ones(dim)
        self.diffusion.coefficients = sigma * np.eye(dim)

    def exact_density(self, x0, xt, t0: float, dt: float = 0.0) -> float:
        mu, sigma2 = self.coefficients
        mean_ = x0 + mu * dt
        return norm.pdf(xt.ravel(), loc=mean_.ravel(), scale=np.sqrt(sigma2 * dt))

    def exact_step(self, x, dt, dZ, t=0.0):
        """Simple Brownian motion can be simulated exactly"""
        sig_sq_dt = np.sqrt(self.coefficients[1] * dt)
        return (x.T + self.coefficients[0] * dt + sig_sq_dt * dZ).T


class OrnsteinUhlenbeck(Overdamped):
    r"""
    Model for OU (ornstein-uhlenbeck):
    Parameters: :math:`[\kappa, \mu, \sigma]`

    .. math::
        dX(t) = \mu(X,t)*dt + \sigma(X,t)*dW_t

    where:
        :math:`\mu(X,t)    = \theta - \kappa X`

        :math:`\sigma(X,t) = \sqrt{2 \sigma}`
    """

    _has_exact_density = True

    def __init__(self, theta=0, kappa=1.0, sigma=1.0, dim=1, **kwargs):
        r"""
        Parameters
        ----------

            theta, kappa: float or ndarray of shape (dim,)

            sigma: float or ndarray of shape (dim,dim)
                constant, >0

            dim: int
                Wanted dimension of the model

        """
        super().__init__(Polynomial(1, domain=Domain.Rd(dim)), Constant(domain=Domain.Rd(dim)), dim=dim, **kwargs)
        self.pos_drift.coefficients = np.concatenate([theta * np.eye(dim), -kappa * np.eye(dim)], axis=0)
        self.diffusion.coefficients = sigma * np.eye(dim)

    # TODO: Adapt for multidemnsionnal case
    def exact_density(self, x0: float, xt: float, t0: float, dt: float = 0.0) -> float:
        theta, kappa, sigma = self.coefficients
        mu = -theta / kappa + (x0 + theta / kappa) * np.exp(kappa * dt)
        var = (1 - np.exp(2 * kappa * dt)) * (sigma / (-kappa))
        return norm.pdf(xt.ravel(), loc=mu.ravel(), scale=np.sqrt(var).ravel())

    def exact_step(self, x, dt, dW, t=0.0):
        theta, kappa, sigma = self.coefficients
        mu = -theta / kappa + (x.T + theta / kappa).T * np.exp(kappa * dt)
        var = (1 - np.exp(2 * kappa * dt)) * (sigma / (-kappa))
        return mu + np.sqrt(var) * dW


def OverdampedSplines1D(domain):
    r"""
    Generate defaut model for estimation of overdamped Langevin dynamics.

    Parameters
    -------------

        domain:
            Range of the model. Number of points will be used as number of knots
    """
    return Overdamped(BSplinesFunction(domain), dim=1)
