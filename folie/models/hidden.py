from .overdamped import Overdamped
from .underdamped import Underdamped
import numpy as np
from ..functions import ParametricFunction


# TODO: Either do it as a complete function that deals the parameters or don't inherite from ParametricFunction
class CombineForceFriction:  # (ParametricFunction):
    """
    A composition function for returning f(x)+g(x)*y
    """

    def __init__(self, f, g, dim_x=1, **kwargs):
        self.f = f
        # super().__init__(self.f.output_shape_, dim_x=None)
        self.g = g  # Check if g has correct output shape
        self._dim_sep = dim_x

    def __call__(self, x, *args, **kwargs):
        return self.f(x, *args, **kwargs) + np.einsum("t...h,th-> t...", self.g(x, *args, **kwargs), x[:, self._dim_sep :])

    def grad_x(self, x, *args, **kwargs):
        return self.f.grad_x(x, *args, **kwargs) + np.einsum("t...he,th-> t...e", self.g.grad_x(x, *args, **kwargs), x[:, self._dim_sep :])

    def grad_coeffs(self, x, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        return np.concatenate((self.f.grad_coeffs(x, *args, **kwargs), np.einsum("t...hc,th-> t...c", self.g.grad_coeffs(x, *args, **kwargs), x[:, self._dim_sep :])), axis=-1)


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
        self._dim = self.dim_x + self.dim_h
        self.force = force.resize((self.dim,))
        self.force.dim_x = self.dim_x
        if diffusion is None:
            self.diffusion = force.copy().resize((self.dim, self.dim))
        else:
            self.diffusion = diffusion.resize((self.dim, self.dim))
        self.diffusion.dim_x = self.dim_x
        self.friction = friction.resize((self.dim, self.dim_h))
        self.friction.dim_x = self.dim_x
        self._n_coeffs_diffusion = self.diffusion.size
        self._n_coeffs_friction = self.friction.size
        self.coefficients = np.concatenate((np.zeros(self.force.size), np.ones(self.friction.size), np.eye(self.dim).flatten()))  # TODO: Replace by a fit with zeros and ones
        self.meandispl = CombineForceFriction(self.force, self.friction, self.dim_x)

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
    TODO: A class that implement an underdamped model with some extras hidden variables linearly correlated with visible ones
    """

    def __init__(self, force, friction, diffusion, dim=1, dim_h=0, **kwargs):
        super().__init__(force, diffusion, dim=dim + dim_h)
        self.dim_h = dim_h
        self.dim_x = dim
