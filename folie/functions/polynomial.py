import numpy as np
from .base import FunctionFromBasis
from ..data import stats_from_input_data


class Constant(FunctionFromBasis):
    """
    A function that return a constant value
    """

    def __init__(self, output_shape=()):
        super().__init__(output_shape)

    def fit(self, X=None, y=None):
        xstats = stats_from_input_data(X)
        self.input_dim_ = xstats.dim
        self.n_basis_features_ = self.input_dim_
        self.coefficients = np.ones((self.n_basis_features_, self.output_size_))
        return self

    def differentiate(self):
        fun = self.copy()  # Inclure extra dim pour la differentiation
        fun._coefficients = np.zeros_like(self._coefficients)  # Il faut aussi freeze les coefficients
        return fun

    def transform(self, x, **kwargs):
        return np.dot(np.ones_like(x), self._coefficients)

    def grad_x(self, x, **kwargs):
        return np.zeros((x.shape[0], *self.output_shape_, x.shape[1]))

    def grad_coeffs(self, x, **kwargs):
        grad_coeffs = np.eye(self.size).reshape(self.n_basis_features_, *self.output_shape_, self.size)
        return np.tensordot(np.ones_like(x), grad_coeffs, axes=1)


class Linear(FunctionFromBasis):
    """
    The linear function f(x) = c x
    """

    def __init__(self, output_shape=()):
        super().__init__(output_shape)

    def fit(self, X=None, y=None):
        xstats = stats_from_input_data(X)
        self.input_dim_ = xstats.dim
        self.n_basis_features_ = self.input_dim_
        self.coefficients = np.ones((self.n_basis_features_, self.output_size_))
        return self

    def differentiate(self):
        fun = Constant(self.output_shape)
        fun.coefficients = self.coefficients.copy()
        return fun

    def transform(self, x, **kwargs):
        return x @ self._coefficients

    def grad_x(self, x, **kwargs):
        len, dim = x.shape
        x_grad = np.ones((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients).reshape(-1, *self.output_shape_, dim)

    def hessian_x(self, x, **kwargs):
        len, dim = x.shape
        x_grad = np.zeros((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients).reshape(-1, *self.output_shape_, dim)

    def grad_coeffs(self, x, **kwargs):
        grad_coeffs = np.eye(self.size).reshape(self.n_basis_features_, *self.output_shape_, self.size)
        return np.tensordot(x, grad_coeffs, axes=1)


class Polynomial(FunctionFromBasis):
    """
    The polynome function
    """

    def __init__(self, deg=1, polynom=np.polynomial.Polynomial, output_shape=()):
        super().__init__(output_shape)
        self.degree = deg + 1
        self.polynom = polynom

    def fit(self, X=None, y=None):
        xstats = stats_from_input_data(X)
        self.input_dim_ = xstats.dim
        self.n_basis_features_ = self.input_dim_ * self.degree
        self.coefficients = np.ones((self.n_basis_features_, self.output_size_))
        return self

    def basis(self, X):
        nsamples, dim = X.shape

        features = np.zeros((nsamples, dim * self.degree))
        for n in range(0, self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            features[:, istart:iend] = self.polynom.basis(n)(X)
        return features

    def transform(self, x, **kwargs):
        return x @ self._coefficients

    def grad_x(self, x, **kwargs):
        len, dim = x.shape
        x_grad = np.ones((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients).reshape(-1, *self.output_shape_, dim)

    def grad_coeffs(self, x, **kwargs):
        grad_coeffs = np.eye(self.size).reshape(self.n_basis_features_, *self.output_shape_, self.size)
        return np.tensordot(x, grad_coeffs, axes=1)
