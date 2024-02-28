import numpy as np

from .overdamped import Overdamped
from ..functions import Constant, Polynomial


class CombineForceFriction:
    """
    A composition function for returning f(x)+g(x)*v
    """

    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, x, v, *args, **kwargs):
        fx = self.model.force(x, *args, **kwargs)
        return fx + np.einsum("t...h,th-> t...", self.model.friction(x, *args, **kwargs).reshape((*fx.shape, v.shape[1])), v)

    def grad_x(self, x, v, *args, **kwargs):
        dfx = self.model.force.grad_x(x, *args, **kwargs)
        return dfx + np.einsum("t...he,th-> t...e", self.model.friction.grad_x(x, *args, **kwargs).reshape((*dfx.shape[:-1], v.shape[1], dfx.shape[-1])), v)

    def hessian_x(self, x, v, *args, **kwargs):
        ddfx = self.model.force.hessian_x(x, *args, **kwargs)
        return ddfx + np.einsum("t...hef,th-> t...ef", self.model.friction.hessian_x(x, *args, **kwargs).reshape((*ddfx.shape[:-2], v.shape[1], *ddfx.shape[-2:])), v)

    def grad_coeffs(self, x, v, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        dfx = self.model.force.grad_coeffs(x, *args, **kwargs)
        return np.concatenate((dfx, np.einsum("t...hc,th-> t...c", self.model.friction.grad_coeffs(x, *args, **kwargs).reshape((*dfx.shape[:-1], v.shape[1], -1)), v)), axis=-1)


class Underdamped(Overdamped):
    def __init__(self, force, friction, diffusion, dim=1, **kwargs):
        """
        Base model for underdamped Langevin equations, defined by

        dX(t) = V(t)

        dV(t) = f(X,t)dt+ gamma(X,t)V(t)dt + sigma(X,t)dW_t

        """
        super().__init__(force, diffusion, dim=dim)
        self.meandispl = CombineForceFriction(self)
        self.friction = friction.resize(self.diffusion.shape)
        if not self.friction.fitted_ and not kwargs.get("friction_is_fitted", False):
            loc_dim = self.dim if self.dim > 0 else 1
            X = np.linspace([-1] * loc_dim, [1] * loc_dim, 5)
            self.friction.fit(X, np.ones((5, *self.diffusion.shape)))

    @Overdamped.dim.setter
    def dim(self, dim):
        self._dim = dim
        if dim >= 1:
            force_shape = (dim,)
            diffusion_shape = (dim, dim)
        else:
            force_shape = ()
            diffusion_shape = ()
        self.force = self.force.resize(force_shape)
        self.diffusion = self.diffusion.resize(diffusion_shape)
        self.friction = self.friction.resize(diffusion_shape)

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.force.coefficients.ravel(), self.diffusion.coefficients.ravel(), self.friction.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.force.coefficients = vals.ravel()[: self.force.size]
        self.diffusion.coefficients = vals.ravel()[self.force.size : self.force.size + self.diffusion.size]
        self.friction.coefficients = vals.ravel()[self.force.size + self.diffusion.size :]

    @property
    def coefficients_friction(self):
        return self.friction.coefficients

    @coefficients_friction.setter
    def coefficients_friction(self, vals):
        self.friction.coefficients = vals


class UnderdampedOrnsteinUhlenbeck(Underdamped):
    """
    Model for OU (ornstein-uhlenbeck):
    Parameters: [kappa, mu, sigma]

    dX(t) = mu(X,t)*dt + sigma(X,t)*dW_t

    where:
        mu(X,t)    = theta - kappa* X
        sigma(X,t) = sqrt(sigma)
    """

    dim = 1
    _has_exact_density = True

    def __init__(self, theta=0, kappa=1.0, sigma=1.0, **kwargs):
        # Init by passing functions to the model
        X = np.linspace(-1, 1, 5).reshape(-1, 1)
        super().__init__(Polynomial(1).fit(X, -1 * np.linspace(-1, 1, 5)), Constant().fit(X, np.ones(5)), Constant().fit(X, np.ones(5)), dim=0, **kwargs)
        self.force.coefficients = np.asarray([theta, -kappa])
        self.friction.coefficients = np.asanyarray(sigma)
        self.diffusion.coefficients = np.asarray(sigma)
