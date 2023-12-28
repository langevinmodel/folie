import numpy as np
from .base import FunctionFromBasis


class Constant(FunctionFromBasis):
    """
    A function that return a constant value
    """

    def __init__(self, output_shape=()):
        super().__init__(output_shape)

    def fit(self, x, y=None):
        _, dim = x.shape
        return self

    def transform(self, x, **kwargs):
        return np.einsum("...d,ld->l...", self._coefficients, np.ones_like(x))

    def grad_x(self, x, **kwargs):
        return np.zeros((x.shape[0],) + self.output_shape_ + (x.shape[1],))

    def grad_coeffs(self, x, **kwargs):
        return np.ones((x.shape[0],) + self.output_shape_ + (1,))


class Linear(FunctionFromBasis):
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
        return np.einsum("...d,ld->l...", self._coefficients, x)

    def grad_x(self, x, **kwargs):
        return np.zeros_like(x)

    def grad_coeffs(self, x, **kwargs):
        return np.ones_like(x)
