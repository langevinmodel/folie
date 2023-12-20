"""
The code in this file is adapted from deeptime (https://github.com/deeptime-ml/deeptime/blob/main/deeptime/base.py)
"""

import numpy as np
from .base import Function


class Constant(Function):
    """
    A function that return a constant value
    """

    def __init__(self, output_shape=()):
        super().__init__(output_shape)

    def fit(self, x, y=None):
        _, dim = x.shape
        return self

    def transform(self, x, **kwargs):
        return self.coefficients * np.ones_like(x)

    def grad_x(self, x, **kwargs):
        return np.zeros_like(x)

    def grad_coeffs(self, x, **kwargs):
        return np.ones_like(x)

    def zero(self):
        r"""Set the coefficients to evaluate the function to zero."""
        self._coefficients = np.zeros((1,) + self.output_shape_)
        return self

    def one(self):
        r"""Get the coefficients to evaluate the function to one"""
        self._coefficients = np.ones((1,) + self.output_shape_)
        return self

    @property
    def is_linear(self) -> bool:
        """Return True is the model is linear in its parameters"""
        return True


class Linear(Function):
    """
    The linear function f(x) = c x
    """

    def __init__(self, output_shape=()):
        super().__init__(output_shape)

    def fit(self, x, y=None):
        _, dim = x.shape
        self.define_output_shape(dim)
        return self

    def transform(self, x, **kwargs):
        return self.coefficients * np.ones_like(x)

    def grad_x(self, x, **kwargs):
        return np.zeros_like(x)

    def grad_coeffs(self, x, **kwargs):
        return np.ones_like(x)

    def zero(self):
        r"""Set the coefficients to evaluate the function to zero."""
        self._coefficients = np.zeros((1,) + self.output_shape_)
        return self

    def one(self):
        r"""Get the coefficients to evaluate the function to one"""
        self._coefficients = np.ones((1,) + self.output_shape_)
        return self

    @property
    def is_linear(self) -> bool:
        """Return True is the model is linear in its parameters"""
        return True
