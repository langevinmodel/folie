import numpy as np
from .base import FunctionFromBasis
from ..data import Trajectories, traj_stats


class Constant(FunctionFromBasis):
    """
    A function that return a constant value
    """

    def __init__(self, output_shape=()):
        super().__init__(output_shape)

    def fit(self, x, y=None):
        if isinstance(x, Trajectories):
            xstats = x.stats
        else:
            xstats = traj_stats(x)
        dim = xstats.dim
        self.n_basis_features_ = dim
        self.coefficients = np.ones((self.n_basis_features_, self.output_size_))
        return self

    def transform(self, x, **kwargs):
        return np.dot(np.ones_like(x), self._coefficients)

    def grad_x(self, x, **kwargs):
        return np.zeros((x.shape[0], *self.output_shape_, x.shape[1]))

    def grad_coeffs(self, x, **kwargs):
        return np.ones((x.shape[0], *self.output_shape_, self.size))


class Linear(FunctionFromBasis):
    """
    The linear function f(x) = c x
    """

    def __init__(self, output_shape=()):
        super().__init__(output_shape)

    def fit(self, x, y=None):
        if isinstance(x, Trajectories):
            xstats = x.stats
        else:
            xstats = traj_stats(x)
        dim = xstats.dim
        self.n_basis_features_ = dim
        self.coefficients = np.ones((self.n_basis_features_, self.output_size_))
        return self

    def transform(self, x, **kwargs):
        return x @ self._coefficients

    def grad_x(self, x, **kwargs):
        len, dim = x.shape
        x_grad = np.ones((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients).reshape(-1, *self.output_shape_, dim)

    def grad_coeffs(self, x, **kwargs):
        grad_coeffs = np.eye(self.size).reshape(self.n_basis_features_, *self.output_shape_, self.size)
        return np.tensordot(x, grad_coeffs, axes=1)
