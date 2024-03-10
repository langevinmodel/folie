from .._numpy import np
import warnings
from scipy.stats import norm

from ..base import Model
from ..functions import Constant, Polynomial, BSplinesFunction, FunctionOffset, FunctionOffsetWithCoefficient, ParametricFunction, ModelOverlay
from ..domains import Domain


class BaseModelOverdamped(Model):
    _has_exact_density = False

    def __init__(self, dim=1, **kwargs):
        """
        Base model for overdamped Langevin equations, defined by

        dX(t) = mu(X,t)dt + sigma(X,t)dW_t

        """

        self._dim = dim
        self.is_biased = False

        if self.dim <= 1:
            output_shape_force = ()
            output_shape_diff = ()
        else:
            output_shape_force = (self.dim,)
            output_shape_diff = (self.dim, self.dim)

        if hasattr(self, "_force") and hasattr(self, "_diffusion"):
            self.force = ModelOverlay(self, "force", output_shape=output_shape_force)
            self.diffusion = ModelOverlay(self, "diffusion", output_shape=output_shape_diff)

        self.meandispl = ModelOverlay(self, "meandispl", output_shape=output_shape_force)

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

    def _meandispl(self, x, *args, **kwargs):
        return self.force(x, *args, **kwargs)

    def meandispl_x(self, x, *args, **kwargs):
        return self.force.grad_x(x, *args, **kwargs)

    def meandispl_xx(self, x, *args, **kwargs):
        return self.force.hessian_x(x, *args, **kwargs)

    def meandispl_coeffs(self, x, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        return self.force.grad_coeffs(x, *args, **kwargs)

    @property
    def coefficients_meandispl(self):
        """Access the coefficients"""
        return self.force.coefficients

    @coefficients_meandispl.setter
    def coefficients_meandispl(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.force.coefficients = vals

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


class Overdamped(BaseModelOverdamped):
    """
    A class that implement a overdamped model with given functions for space dependency
    """

    def __init__(self, force, diffusion=None, dim=None, has_bias=None, **kwargs):
        r"""
        Initialize an overdamped Langevin model

        Parameters
        ----------
        force, diffusion : Functions
            Functions for the spatial dependance of the force and position.
            If diffusion is not given it default to the copy of force
        dim : int
            Dimension of the model. By default it is the dimension of the domain of the force
        has_bias: None, bool or ParametricFunction
            If None, assume no bias in the data.
            If true, this assume that an extra column is present in the data
            If this is a ParametricFunction, the bias is the g(x)*f_bias with g(x) a function to optimize

        """
        if dim is None:
            dim = force.domain.dim
        super().__init__(dim=dim)
        if dim > 1:
            force_shape = (dim,)
            diffusion_shape = (dim, dim)
        else:
            force_shape = ()
            diffusion_shape = ()
        self.force = force.resize(force_shape)
        if diffusion is None:
            self.diffusion = force.copy().resize(diffusion_shape)
        else:
            self.diffusion = diffusion.resize(diffusion_shape)
        # loc_dim = dim if dim > 0 else 1
        # X = np.linspace([-1] * loc_dim, [1] * loc_dim, 5)  # TODO: Ne marche pas si on a un mesh
        # if not self.force.fitted_ and not kwargs.get("force_is_fitted", False):
        #     self.force.fit(X)
        # if not self.diffusion.fitted_ and not kwargs.get("diffusion_is_fitted", False):

        #     diff_target = np.concatenate([np.eye(loc_dim)[None, ...]] * 5).reshape(5, *diffusion_shape)
        #     self.diffusion.fit(X, diff_target)
        if has_bias is not None:
            self.add_bias(has_bias)

        # TODO: Ensuite il faut trouver le domain du modèle, et vérifier s'il sont tous compatible

    @BaseModelOverdamped.dim.setter
    def dim(self, dim):
        if dim > 1:
            force_shape = (dim,)
            diffusion_shape = (dim, dim)
        else:
            force_shape = ()
            diffusion_shape = ()
        self.force = self.force.resize(force_shape)
        self.diffusion = self.diffusion.resize(diffusion_shape)
        self._dim = dim

    def preprocess_traj(self, trj, **kwargs):
        if hasattr(self.force.domain, "localize_data") or hasattr(self.diffusion.domain, "localize_data"):
            # Check if domain are compatible
            cells_idx, loc_x = self.force.domain.localize_data(trj["x"], **kwargs)
            trj["cells_idx"] = cells_idx
            trj["loc_x"] = loc_x
        return trj

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.force.coefficients.ravel(), self.diffusion.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.force.coefficients = vals.ravel()[: self.force.size]
        self.diffusion.coefficients = vals.ravel()[self.force.size : self.force.size + self.diffusion.size]

    @property
    def coefficients_force(self):
        return self.force.coefficients

    @coefficients_force.setter
    def coefficients_force(self, vals):
        self.force.coefficients = vals

    @property
    def coefficients_diffusion(self):
        return self.diffusion.coefficients

    @coefficients_diffusion.setter
    def coefficients_diffusion(self, vals):
        self.diffusion.coefficients = vals

    def add_bias(self, bias=True):
        if isinstance(bias, bool) and bias:
            self.force = FunctionOffset(self.force, self.diffusion)
        elif isinstance(bias, ParametricFunction):
            self.force = FunctionOffsetWithCoefficient(self.force, bias)
        self.is_biased = True

    def remove_bias(self):
        if self.is_biased:
            self.force = self.force.f
            self.is_biased = False
        else:
            print("Model is not biased")


#  Set of quick interface to more common models


class BrownianMotion(Overdamped):
    """
    Model for (forced) Brownian Motion
    Parameters:  [mu, sigma]

    dX(t) = mu(X,t)dt + sigma(X,t)dW_t

    where:
        mu(X,t)    = mu   (constant)
        sigma(X,t) = sqrt(sigma)   (constant, >0)
    """

    _has_exact_density = True

    def __init__(self, mu=0, sigma=1.0, dim=1, **kwargs):
        super().__init__(Constant(domain=Domain.Rd(dim)), Constant(domain=Domain.Rd(dim)), dim=dim, **kwargs)
        self.force.coefficients = mu * np.ones(dim)
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
    """
    Model for OU (ornstein-uhlenbeck):
    Parameters: [kappa, mu, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = theta - kappa* X
        sigma(X,t) = sqrt(sigma)
    """

    _has_exact_density = True

    def __init__(self, theta=0, kappa=1.0, sigma=1.0, dim=1, **kwargs):
        # Init by passing functions to the model
        # X = np.linspace(-1, 1, 5).reshape(-1, 1)
        # .fit(X, -1 * np.linspace(-1, 1, 5))
        # .fit(X, np.ones(5))
        super().__init__(Polynomial(1, domain=Domain.Rd(dim)), Constant(domain=Domain.Rd(dim)), dim=dim, **kwargs)
        self.force.coefficients = np.concatenate([theta * np.eye(dim), -kappa * np.eye(dim)], axis=0)
        self.diffusion.coefficients = sigma * np.eye(dim)

    # TODO: Adapt for multidemnsionnal case
    def exact_density(self, x0: float, xt: float, t0: float, dt: float = 0.0) -> float:
        theta, kappa, sigma = self.coefficients
        mu = -theta / kappa + (x0 + theta / kappa) * np.exp(kappa * dt)
        var = (1 - np.exp(2 * kappa * dt)) * (sigma / (-2 * kappa))
        return norm.pdf(xt.ravel(), loc=mu.ravel(), scale=np.sqrt(var).ravel())

    def exact_step(self, x, dt, dZ, t=0.0):
        theta, kappa, sigma = self.coefficients
        mu = -theta / kappa + (x.T + theta / kappa).T * np.exp(kappa * dt)
        var = (1 - np.exp(2 * kappa * dt)) * (sigma / (-2 * kappa))
        return mu * dt + np.sqrt(var * dt) * dZ


def OverdampedSplines1D(domain):
    """
    Generate defaut model for estimation of overdamped Langevin dynamics.

    Parameters
    -------------

        knots: int of array
            Either the number of knots to use in the spline or a list of knots.
            The more knots the more precise the model but it will be more expensive to run and require more data for precise estimation.
    """
    return Overdamped(BSplinesFunction(domain), dim=1)
