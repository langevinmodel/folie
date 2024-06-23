from .overdamped import Overdamped
from .underdamped import Underdamped
from .._numpy import np


class OverdampedHidden(Overdamped):
    """
    A class that implement an overdamped model with some extras hidden variables linearly correlated with visible ones


    d \\begin{pmatrix} X(t) \\ h(t) \\end{pmatrix} = f(X,t)dt+ gamma(X,t)h(t)dt + sigma(X,t)dW_t

    where

    f(X,t) is a $dim_x + dim_h$ vector

    gamma(X,t) is a $(dim_x + dim_h) \times dim_h$ matrix

    sigma(X,t) is a $(dim_x + dim_h) \times (dim_x + dim_h)$ matrix

    """

    def __init__(self, pos_drift, friction, diffusion, dim=1, dim_h=0, **kwargs):
        self.dim_h = dim_h
        self.dim_x = dim
        pos_drift.dim_x = self.dim_x
        diffusion.dim_x = self.dim_x
        friction.dim_x = self.dim_x

        super().__init__(pos_drift, diffusion, dim=self.dim_x + self.dim_h, **kwargs)
        self.friction = friction.resize((self.dim, self.dim_h))

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.pos_drift.coefficients.ravel(), self.friction.coefficients.ravel(), self.diffusion.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals.ravel()[: self.pos_drift.size]
        self.friction.coefficients = vals.ravel()[self.pos_drift.size : self.pos_drift.size + self.friction.size]
        self.diffusion.coefficients = vals.ravel()[self.pos_drift.size + self.friction.size :]

    @property
    def coefficients_friction(self):
        return self.friction.coefficients

    @coefficients_friction.setter
    def coefficients_friction(self, vals):
        self.friction.coefficients = vals

    def _drift(self, x, *args, **kwargs):
        return self.pos_drift(x, *args, **kwargs) + np.einsum("t...h,th-> t...", self.friction(x, *args, **kwargs), x[:, self.dim_x :])

    def _drift_dx(self, x, *args, **kwargs):
        return self.pos_drift.grad_x(x, *args, **kwargs) + np.einsum("t...he,th-> t...e", self.friction.grad_x(x, *args, **kwargs), x[:, self.dim_x :])

    def _drift_d2x(self, x, *args, **kwargs):
        return self.pos_drift.hessian_x(x, *args, **kwargs) + np.einsum("t...hef,th-> t...ef", self.friction.hessian_x(x, *args, **kwargs), x[:, self.dim_x :])

    def _drift_dcoeffs(self, x, *args, **kwargs):
        """
        Jacobian of the drift with respect to coefficients
        """
        return np.concatenate((self.pos_drift.grad_coeffs(x, *args, **kwargs), np.einsum("t...hc,th-> t...c", self.friction.grad_coeffs(x, *args, **kwargs), x[:, self.dim_x :])), axis=-1)

    @property
    def coefficients_drift(self):
        """Access the coefficients"""
        return np.concatenate((self.pos_drift.coefficients.ravel(), self.friction.coefficients.ravel()))

    @coefficients_drift.setter
    def coefficients_drift(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals.ravel()[: self.pos_drift.size]
        self.friction.coefficients = vals.ravel()[self.pos_drift.size : self.pos_drift.size + self.friction.size]


class UnderdampedHidden(Underdamped):
    """
    A class that implement an underdamped model with some extras hidden variables linearly correlated with visible ones
    """

    def __init__(self, pos_drift, friction, diffusion, dim=1, dim_h=0, **kwargs):
        self.dim_h = dim_h
        self.dim_x = dim
        # TODO: le seul truc à changer c'est peut-être la force sur les variables cachées pour pad avec des zéros
        # Force = encapsulated(force)

        pos_drift.dim_x = self.dim_x
        diffusion.dim_x = self.dim_x
        friction.dim_x = self.dim_x
        super().__init__(pos_drift, friction, diffusion, dim=self.dim_x + self.dim_h, **kwargs)
