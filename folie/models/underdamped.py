from .._numpy import np

from .overdamped import Overdamped
from ..functions import Constant, Polynomial


class Underdamped(Overdamped):
    """
    Base model for underdamped Langevin equations, defined by

    .. math ::

    dX(t) = V(t)

    dV(t) = f(X,t)dt+ gamma(X,t)V(t)dt + sigma(X,t)dW_t

    """

    def __init__(self, force, friction, diffusion, dim=1, **kwargs):

        if friction is diffusion:
            friction = diffusion.copy()
        super().__init__(force, diffusion, dim=dim)
        self.friction = friction.resize(self.diffusion.shape)

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

    def _meandispl(self, x, v, *args, **kwargs):
        fx = self.force(x, *args, **kwargs)
        return fx - np.einsum("t...h,th-> t...", self.friction(x, *args, **kwargs).reshape((*fx.shape, v.shape[1])), v)

    def _meandispl_dx(self, x, v, *args, **kwargs):
        dfx = self.force.grad_x(x, *args, **kwargs)
        return dfx - np.einsum("t...he,th-> t...e", self.friction.grad_x(x, *args, **kwargs).reshape((*dfx.shape[:-1], v.shape[1], dfx.shape[-1])), v)

    def _meandispl_d2x(self, x, v, *args, **kwargs):
        ddfx = self.force.hessian_x(x, *args, **kwargs)
        return ddfx - np.einsum("t...hef,th-> t...ef", self.friction.hessian_x(x, *args, **kwargs).reshape((*ddfx.shape[:-2], v.shape[1], *ddfx.shape[-2:])), v)

    def _meandispl_dcoeffs(self, x, v, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        dfx = self.force.grad_coeffs(x, *args, **kwargs)
        return np.concatenate((dfx, -1 * np.einsum("t...hc,th-> t...c", self.friction.grad_coeffs(x, *args, **kwargs).reshape((*dfx.shape[:-1], v.shape[1], -1)), v)), axis=-1)

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.force.coefficients.ravel(), self.friction.coefficients.ravel(), self.diffusion.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.force.coefficients = vals.ravel()[: self.force.size]
        self.diffusion.coefficients = vals.ravel()[self.force.size + self.friction.size :]
        self.friction.coefficients = vals.ravel()[self.force.size : self.force.size + self.friction.size]

    @property
    def coefficients_meandispl(self):
        """Access the coefficients"""
        return np.concatenate((self.force.coefficients.ravel(), self.friction.coefficients.ravel()))

    @coefficients_meandispl.setter
    def coefficients_meandispl(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.force.coefficients = vals.ravel()[: self.force.size]
        self.friction.coefficients = vals.ravel()[self.force.size : self.force.size + self.friction.size]


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
        # TODO: Update to not use fit
        super().__init__(Polynomial(1), Constant(), Constant(), dim=1, **kwargs)
        self.force.coefficients = np.asarray([theta, -kappa])
        self.friction.coefficients = np.asanyarray(sigma)
        self.diffusion.coefficients = np.asarray(sigma)
