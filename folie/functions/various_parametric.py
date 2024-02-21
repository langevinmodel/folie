import numpy as np

from .base import ParametricFunction


class FunctionFromBasis(ParametricFunction):
    """
    Encaspulate function basis from VolterraBasis
    """

    def __init__(self, output_shape=(), basis=None):
        super().__init__(output_shape)
        self.basis = basis

    def fit(self, x, y=None, **kwargs):
        if self.basis is not None:
            self.basis.fit(x, y, **kwargs)
            self.n_functions_features_ = self.basis.n_output_features_
        super.fit(x, y)
        self.coefficients = np.zeros((self.n_functions_features_, self.output_size_))
        return self

    def transform(self, x, *args, **kwargs):
        return self.basis(x) @ self._coefficients

    def grad_x(self, x, **kwargs):
        _, dim = x.shape
        return np.einsum("nbd,bs->nsd", self.basis.deriv(x), self._coefficients).reshape(-1, *self.output_shape_, dim)

    def grad_coeffs(self, x, **kwargs):
        grad_coeffs = np.eye(self.size).reshape(self.n_functions_features_, *self.output_shape_, self.size)
        return np.tensordot(x, grad_coeffs, axes=1)

    def gram(self, x):
        """
        Compute gram matrix on points x
        """
        basis_vals = self.basis(x).reshape(x.shape[0], -1)
        return np.dot(basis_vals.T, basis_vals)


class sklearnTransformer(ParametricFunction):
    """
    take any sklearn transformer and build a fonction from it
    """

    def __init__(self, transformer, output_shape=(), coefficients=None):
        super().__init__(output_shape, coefficients)
        self.transformer = transformer

    def fit(self, X, y=None, **kwargs):
        self.transformer = self.transformer.fit(X)
        self.n_functions_features_ = self.transformer.transform(X[:5, :]).shape[1]
        super().fit(X, y, **kwargs)
        return self

    def transform(self, x, *args, **kwargs):
        return self.transformer.transform(x) @ self._coefficients

    def grad_coeffs(self, x, **kwargs):
        grad_coeffs = np.eye(self.size).reshape(self.n_functions_features_, *self.output_shape_, self.size)
        return np.tensordot(self.transformer.transform(x), grad_coeffs, axes=1)


class FreezeCoefficients(ParametricFunction):
    r"""
    Function that only expose a subset of the coefficients of the underlying function
    """

    def __init__(self, f, freezed_coefficients):
        self.f = f

    def fit(self, x, y=None, **kwargs):
        self.f.fit(x, y, **kwargs)

        return self

    def transform(self, x, *args, **kwargs):
        r"""Transforms the input data."""
        return self.f.transform(x)

    def grad_x(self, x, **kwargs):
        r"""Gradient of the function with respect to input data"""

        return self.f.grad_x(x)

    def hessian_x(self, x, **kwargs):
        """
        Hessian of the function with respect to input data
        """
        return self.f.hessian_x(x)

    def grad_coeffs(self, x, **kwargs):
        r"""Transforms the input data."""


class FunctionOffsetWithCoefficient(ParametricFunction):
    """
    A composition function for returning f(x)+g(x)*y
    """

    def __init__(self, f, g=None, output_shape=(), **kwargs):
        self.f = f
        self.g = g
        # Check if g is a Function or a constant
        super().__init__(self.f.output_shape_)

    def fit(self, X, y=None, **kwargs):
        # First fit sub function with representative array then fit the global one
        pass

    def transform(self, x, v):
        return self.f(x) + np.einsum("t...h,th-> t...", self.g(x), v)

    def grad_x(self, x, v):
        return self.f.grad_x(x) + np.einsum("t...he,th-> t...e", self.g.grad_x(x), v)

    def grad_coeffs(self, x, v, t: float = 0.0):
        """
        Jacobian of the force with respect to coefficients
        """
        return np.concatenate((self._force.grad_coeffs(x), np.einsum("t...hc,th-> t...c", self._friction.grad_coeffs(x[:, : self.dim_x]), v)), axis=-1)
