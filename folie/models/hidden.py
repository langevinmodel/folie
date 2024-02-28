from .overdamped import Overdamped
from .underdamped import Underdamped
import numpy as np


class CombineForceFrictionHidden:
    """
    A composition function for returning f(x)+g(x)*y
    """

    def __init__(self, model, dim_x=1, **kwargs):
        self.model = model
        self._dim_sep = dim_x

    def __call__(self, x, *args, **kwargs):
        return self.model.force(x, *args, **kwargs) + np.einsum("t...h,th-> t...", self.model.friction(x, *args, **kwargs), x[:, self._dim_sep :])

    def grad_x(self, x, *args, **kwargs):
        return self.model.force.grad_x(x, *args, **kwargs) + np.einsum("t...he,th-> t...e", self.model.friction.grad_x(x, *args, **kwargs), x[:, self._dim_sep :])

    def hessian_x(self, x, *args, **kwargs):
        return self.model.force.hessian_x(x, *args, **kwargs) + np.einsum("t...hef,th-> t...ef", self.model.friction.hessian_x(x, *args, **kwargs), x[:, self._dim_sep :])

    def grad_coeffs(self, x, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        return np.concatenate((self.model.force.grad_coeffs(x, *args, **kwargs), np.einsum("t...hc,th-> t...c", self.model.friction.grad_coeffs(x, *args, **kwargs), x[:, self._dim_sep :])), axis=-1)


class OverdampedHidden(Overdamped):
    """
    A class that implement an overdamped model with some extras hidden variables linearly correlated with visible ones


    d \\begin{pmatrix} X(t) \\ h(t) \\end{pmatrix} = f(X,t)dt+ gamma(X,t)h(t)dt + sigma(X,t)dW_t

    where

    f(X,t) is a $dim_x + dim_h$ vector

    gamma(X,t) is a $(dim_x + dim_h) \times dim_h$ matrix

    sigma(X,t) is a $(dim_x + dim_h) \times (dim_x + dim_h)$ matrix

    """

    def __init__(self, force, friction, diffusion, dim=1, dim_h=0, **kwargs):
        self.dim_h = dim_h
        self.dim_x = dim
        force.dim_x = self.dim_x
        diffusion.dim_x = self.dim_x
        friction.dim_x = self.dim_x

        super().__init__(force, diffusion, dim=self.dim_x + self.dim_h, **kwargs)
        self.friction = friction.resize((self.dim, self.dim_h))
        if not self.friction.fitted_ and not kwargs.get("friction_is_fitted", False):  # Set friction to constant one
            loc_dim = self.dim if self.dim > 0 else 1
            X = np.linspace([-1] * loc_dim, [1] * loc_dim, 5)
            self.friction.fit(X, np.ones((5, self.dim, self.dim_h)))

        self.meandispl = CombineForceFrictionHidden(self, self.dim_x)

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.force.coefficients.ravel(), self.friction.coefficients.ravel(), self.diffusion.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.force.coefficients = vals.ravel()[: self.force.size]
        self.friction.coefficients = vals.ravel()[self.force.size : self.force.size + self.friction.size]
        self.diffusion.coefficients = vals.ravel()[self.force.size + self.friction.size :]

    @property
    def coefficients_friction(self):
        return self.friction.coefficients

    @coefficients_friction.setter
    def coefficients_friction(self, vals):
        self.friction.coefficients = vals


class UnderdampedHidden(Underdamped):
    """
    A class that implement an underdamped model with some extras hidden variables linearly correlated with visible ones
    """

    def __init__(self, force, friction, diffusion, dim=1, dim_h=0, **kwargs):
        self.dim_h = dim_h
        self.dim_x = dim
        # TODO: le seul truc à changer c'est peut-être la force sur les variables cachées pour pad avec des zéros
        # Force = encapsulated(force)

        force.dim_x = self.dim_x
        diffusion.dim_x = self.dim_x
        friction.dim_x = self.dim_x
        super().__init__(force, friction, diffusion, dim=self.dim_x + self.dim_h, **kwargs)
