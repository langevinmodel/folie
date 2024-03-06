from .._numpy import np
from .base import ParametricFunction
from ..domains import Domain


class Constant(ParametricFunction):
    """
    A function that return a constant value
    """

    def __init__(self, domain=None, output_shape=(), coefficients=None):
        if domain is None:
            domain = Domain.Rd(dim=1)
        super().__init__(domain, output_shape, coefficients)
        self.n_functions_features_ = 1

    def differentiate(self):
        fun = self.copy()  # Inclure extra dim pour la differentiation
        fun._coefficients = np.zeros_like(self._coefficients)  # Il faut aussi freeze les coefficients
        return fun

    def transform(self, x, *args, **kwargs):
        return np.ones((x.shape[0], 1)) * self._coefficients

    def transform_x(self, x, *args, **kwargs):
        return np.zeros((x.shape[0], self.output_size_, x.shape[1]))

    def transform_xx(self, x, *args, **kwargs):
        return np.zeros((x.shape[0], self.output_size_, x.shape[1], x.shape[1]))

    def transform_coeffs(self, x, *args, **kwargs):
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        return np.tensordot(np.ones((x.shape[0], 1)), transform_coeffs, axes=1)


class Linear(ParametricFunction):
    """
    The linear function f(x) = c x
    """

    def __init__(self, domain=None, output_shape=(), coefficients=None):
        if domain is None:
            domain = Domain.Rd(dim=1)
        self.n_functions_features_ = domain.dim
        super().__init__(domain, output_shape, coefficients)

    def differentiate(self):
        fun = Constant(self.output_shape)
        fun.coefficients = self.coefficients.copy()
        return fun

    def transform(self, x, *args, **kwargs):
        return x @ self._coefficients

    def transform_x(self, x, *args, **kwargs):
        len, dim = x.shape
        x_grad = np.ones((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients)  # .reshape(-1, *self.output_shape_, dim)

    def transform_xx(self, x, *args, **kwargs):
        return np.zeros((x.shape[0], self.output_size_, x.shape[1], x.shape[1]))

    def hessian_x(self, x, *args, **kwargs):
        len, dim = x.shape
        x_grad = np.zeros((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients)  # .reshape(-1, *self.output_shape_, dim)

    def transform_coeffs(self, x, *args, **kwargs):
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, *self.output_shape_, self.size)
        return np.tensordot(x, transform_coeffs, axes=1)


class Polynomial(ParametricFunction):
    """
    The polynomial function
    """

    def __init__(self, deg=None, polynom=np.polynomial.Polynomial(1), domain=None, output_shape=(), coefficients=None):
        if deg is None:
            deg = polynom.degree()
        self.degree = deg + 1
        self.polynom = polynom
        if domain is None:
            domain = Domain.Rd(dim=1)
        self.n_functions_features_ = domain.dim * self.degree
        super().__init__(output_shape, coefficients)

    def transform(self, x, *args, **kwargs):
        _, dim = x.shape
        res = 0.0
        for n in range(self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            res += self.polynom.basis(n)(x) @ self._coefficients[istart:iend, :]
        return res

    def transform_x(self, x, *args, **kwargs):
        _, dim = x.shape
        res = 0.0
        for n in range(0, self.degree):  # First value is zero anyway
            istart = n * dim
            iend = (n + 1) * dim
            grad = self.polynom.basis(n).deriv(1)(x)[..., None] * np.eye(dim)[None, :, :]
            res += np.einsum("nbd,bs->nsd", grad, self._coefficients[istart:iend, :])  # .reshape(-1, *self.output_shape_, dim)
        return res

    def transform_xx(self, x, *args, **kwargs):
        _, dim = x.shape
        res = 0.0
        for n in range(0, self.degree):  # First values are zero anyway
            istart = n * dim
            iend = (n + 1) * dim
            grad = self.polynom.basis(n).deriv(2)(x)[..., None] * np.eye(dim)[None, :, :]
            res += np.einsum("nbd,bs->nsd", grad, self._coefficients[istart:iend, :])  # .reshape(-1, *self.output_shape_, dim)
        return res

    def transform_coeffs(self, x, *args, **kwargs):
        _, dim = x.shape
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        res = 0.0
        for n in range(self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            res += np.tensordot(self.polynom.basis(n)(x), transform_coeffs[istart:iend, :], axes=1)
        return res
