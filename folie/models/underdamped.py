from .._numpy import np

from .overdamped import Overdamped
from ..functions import Constant, Polynomial, ModelOverlay


class Underdamped(Overdamped):
    """
    Base model for underdamped Langevin equations, defined by

    .. math ::

    dX(t) = V(t)

    dV(t) = f(X,t)dt - gamma(X,t)V(t)dt + sigma(X,t)dW_t

    """

    def __init__(self, force, friction, diffusion, dim=1, FDT=False, **kwargs):

        if friction is diffusion:
            friction = diffusion.copy()
        super().__init__(force, diffusion, dim=dim)
        if not FDT:
            self.friction = friction.resize(self.diffusion.shape)

        if self.dim <= 1:
            output_shape_drift = ()
            output_shape_diff = ()
        else:
            output_shape_drift = (self.dim,)
            output_shape_diff = (self.dim, self.dim)
        self.drift_dx = ModelOverlay(self, "_drift_dx", output_shape=output_shape_drift)
        self.drift_dcoeffs = ModelOverlay(self, "_drift_dcoeffs", output_shape=output_shape_drift)

    @Overdamped.dim.setter
    def dim(self, dim):
        self._dim = dim
        if dim >= 1:
            force_shape = (dim,)
            diffusion_shape = (dim, dim)
        else:
            force_shape = ()
            diffusion_shape = ()
        self.pos_drift = self.pos_drift.resize(force_shape)
        self.diffusion = self.diffusion.resize(diffusion_shape)
        self.friction = self.friction.resize(diffusion_shape)

    def _drift(self, x, v, *args, **kwargs):
        fx = self.pos_drift(x, *args, **kwargs)
        return fx - np.einsum("t...h,th-> t...", self.friction(x, *args, **kwargs).reshape((*fx.shape, v.shape[1])), v)

    def _drift_dx(self, x, v, *args, **kwargs):
        dfx = self.pos_drift.grad_x(x, *args, **kwargs)
        return dfx - np.einsum("t...he,th-> t...e", self.friction.grad_x(x, *args, **kwargs).reshape((*dfx.shape[:-1], v.shape[1], dfx.shape[-1])), v)

    def _drift_d2x(self, x, v, *args, **kwargs):
        ddfx = self.pos_drift.hessian_x(x, *args, **kwargs)
        return ddfx - np.einsum("t...hef,th-> t...ef", self.friction.hessian_x(x, *args, **kwargs).reshape((*ddfx.shape[:-2], v.shape[1], *ddfx.shape[-2:])), v)

    def _drift_dcoeffs(self, x, v, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        dfx = self.pos_drift.grad_coeffs(x, *args, **kwargs)
        return np.concatenate((dfx, -1 * np.einsum("t...hc,th-> t...c", self.friction.grad_coeffs(x, *args, **kwargs).reshape((*dfx.shape[:-1], v.shape[1], -1)), v)), axis=-1)

    def _drift_dx_dcoeffs(self, x, v, *args, **kwargs):
        """
        Computes the derivative with respect to x and the coefficients of the drift:
          drift(x, v) = pos_drift(x) - friction(x) @ v
        It is given by:
          grad_x_dcoeffs[pos_drift]  concatenated with  - einsum( friction.grad_x_dcoeffs, v )
        """
        # pos_term has shape: (t, *output_shape, x_dim, pos_coeff_dim)
        pos_term = self.pos_drift.grad_x_dcoeffs(x, *args, **kwargs)
        # friction.grad_x_dcoeffs originally has shape: (t, *output_shape, x_dim, friction_coeff_dim)
        # We reshape it to insert the v-dimension (which has size v_dim)
        # New shape becomes: (t, *output_shape, x_dim, v_dim, friction_coeff_dim)
        friction_grad = self.friction.grad_x_dcoeffs(x, *args, **kwargs).reshape(
            (*pos_term.shape[:-1], v.shape[1], -1)
        )
        # Contract the inserted v-dimension with v (shape: (t, v_dim))
        # The einsum string:
        #   "t...dhc,th->t...dc"
        # here: d is the x dimension, h is the inserted v-dimension, and c is the friction coefficient index.
        friction_term = np.einsum("t...dhc,th->t...dc", friction_grad, v)
        # Now friction_term has shape: (t, *output_shape, x_dim, friction_coeff_dim)
        # Concatenate along the last axis (the coefficient dimension)
        return np.concatenate((pos_term, -friction_term), axis=-1)
    
    
    def _drift_d2x_dcoeffs(self, x, v, *args, **kwargs):
        """
        Computes the second derivative with respect to x and the derivative with respect to the coefficients
        of the drift function:
          drift(x, v) = pos_drift(x) - friction(x) @ v
        It is given by:
          hessian_x_dcoeffs[pos_drift]  concatenated with  - einsum( friction.hessian_x_dcoeffs, v )
        """
        # pos_term has shape: (t, *output_shape, x_dim, x_dim, pos_coeff_dim)
        pos_term = self.pos_drift.hessian_x_dcoeffs(x, *args, **kwargs)
        # friction.hessian_x_dcoeffs originally has shape: (t, *output_shape, x_dim, x_dim, friction_coeff_dim)
        # We reshape to insert the v-dimension before the last axis.
        # New shape: (t, *output_shape, x_dim, x_dim, v_dim, friction_coeff_dim)
        friction_grad = self.friction.hessian_x_dcoeffs(x, *args, **kwargs).reshape(
            (*pos_term.shape[:-1], v.shape[1], -1)
        )
        # Contract over the inserted v-dimension with v (shape: (t, v_dim)).
        # Using einsum:
        #   "t...dhfc,th->t...dfc"
        # where d and f are the two x-dimensions, h is the inserted v-dimension, and c the friction coefficient index.
        friction_term = np.einsum("t...dhfc,th->t...dfc", friction_grad, v)
        # Concatenate along the coefficient axis (last axis)
        return np.concatenate((pos_term, -friction_term), axis=-1)


    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.pos_drift.coefficients.ravel(), self.friction.coefficients.ravel(), self.diffusion.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals.ravel()[: self.pos_drift.size]
        self.diffusion.coefficients = vals.ravel()[self.pos_drift.size + self.friction.size :]
        self.friction.coefficients = vals.ravel()[self.pos_drift.size : self.pos_drift.size + self.friction.size]

    @property
    def coefficients_drift(self):
        """Access the coefficients"""
        return np.concatenate((self.pos_drift.coefficients.ravel(), self.friction.coefficients.ravel()))

    @coefficients_drift.setter
    def coefficients_drift(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals.ravel()[: self.pos_drift.size]
        self.friction.coefficients = vals.ravel()[self.pos_drift.size : self.pos_drift.size + self.friction.size]

class UnderdampedFDT(Underdamped):
    """
    Base model for underdamped Langevin equations modelling equilibrium systems, defined by

    .. math ::

    dX(t) = V(t)

    dV(t) = f(X,t)dt - gamma(X,t)V(t)dt + sqrt(2mkBTgamma(X,t))dW_t

    """
    def __init__(self, force, diffusion, mass_kBT=1., dim=1, **kwargs):
        super().__init__(force, diffusion, diffusion, dim=dim, FDT=True)
        self.mass_kBT = mass_kBT

    @property
    def friction(self):
        friction = self.diffusion.copy()
        friction.coefficients /= self.mass_kBT
        return friction

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.pos_drift.coefficients.ravel(), self.diffusion.coefficients.ravel(), ))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals.ravel()[: self.pos_drift.size]
        self.diffusion.coefficients = vals.ravel()[self.pos_drift.size :]
        

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
        self.pos_drift.coefficients = np.asarray([theta, -kappa])
        self.friction.coefficients = np.asanyarray(sigma)
        self.diffusion.coefficients = np.asarray(sigma)
