from .._numpy import np
from .base import ParametricFunction
from ..domains import Domain
from numpy.polynomial import Polynomial


class Constant(ParametricFunction):
    """
    A function that return a constant value
    """

    def __init__(self, domain=None, output_shape=(), coefficients=None):
        if domain is None:
            domain = Domain.Rd(dim=1)
        self.n_functions_features_ = 1
        super().__init__(domain, output_shape, coefficients)

    def transform(self, x, *args, **kwargs):
        return np.ones((x.shape[0], 1)) * self._coefficients

    def transform_dx(self, x, *args, **kwargs):
        return np.zeros((x.shape[0], self.output_size_, x.shape[1]))

    def transform_d2x(self, x, *args, **kwargs):
        return np.zeros((x.shape[0], self.output_size_, x.shape[1], x.shape[1]))

    def transform_dcoeffs(self, x, *args, **kwargs):
        transform_dcoeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        return np.tensordot(np.ones((x.shape[0], 1)), transform_dcoeffs, axes=1)


class Linear(ParametricFunction):
    """
    The linear function f(x) = c x
    """

    def __init__(self, domain=None, output_shape=(), coefficients=None):
        if domain is None:
            domain = Domain.Rd(dim=1)
        self.n_functions_features_ = domain.dim
        super().__init__(domain, output_shape, coefficients)

    def transform(self, x, *args, **kwargs):
        return x @ self._coefficients

    def transform_dx(self, x, *args, **kwargs):
        len, dim = x.shape
        x_grad = np.ones((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients)  # .reshape(-1, *self.output_shape_, dim)

    def transform_d2x(self, x, *args, **kwargs):
        return np.zeros((x.shape[0], self.output_size_, x.shape[1], x.shape[1]))

    def hessian_x(self, x, *args, **kwargs):
        len, dim = x.shape
        x_grad = np.zeros((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients)  # .reshape(-1, *self.output_shape_, dim)

    def transform_dcoeffs(self, x, *args, **kwargs):
        transform_dcoeffs = np.eye(self.size).reshape(self.n_functions_features_, *self.output_shape_, self.size)
        return np.tensordot(x, transform_dcoeffs, axes=1)


class Polynomial(ParametricFunction):
    """
    The polynomial function
    """

    def __init__(self, deg=None, polynom=Polynomial(1), domain=None, output_shape=(), coefficients=None):
        if deg is None:
            deg = polynom.degree()
        self.degree = deg + 1
        self.polynom = polynom
        if domain is None:
            domain = Domain.Rd(dim=1)
        self.n_functions_features_ = domain.dim * self.degree
        super().__init__(domain, output_shape, coefficients)

    def transform(self, x, *args, **kwargs):
        _, dim = x.shape
        res = 0.0
        for n in range(self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            res += self.polynom.basis(n)(x) @ self._coefficients[istart:iend, :]
        return res

    def transform_dx(self, x, *args, **kwargs):
        _, dim = x.shape
        res = 0.0
        for n in range(0, self.degree):  # First value is zero anyway
            istart = n * dim
            iend = (n + 1) * dim
            grad = self.polynom.basis(n).deriv(1)(x)[..., None] * np.eye(dim)[None, :, :]
            res += np.einsum("nbd,bs->nsd", grad, self._coefficients[istart:iend, :])  # .reshape(-1, *self.output_shape_, dim)
        return res

    def transform_d2x(self, x, *args, **kwargs):
        _, dim = x.shape
        res = 0.0
        for n in range(0, self.degree):  # First values are zero anyway
            istart = n * dim
            iend = (n + 1) * dim
            grad = self.polynom.basis(n).deriv(2)(x)[..., None] * np.eye(dim)[None, :, :]
            res += np.einsum("nbd,bs->nsd", grad, self._coefficients[istart:iend, :])  # .reshape(-1, *self.output_shape_, dim)
        return res

    def transform_dcoeffs(self, x, *args, **kwargs):
        _, dim = x.shape
        transform_dcoeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        res = 0.0
        for n in range(self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            res += np.tensordot(self.polynom.basis(n)(x), transform_dcoeffs[istart:iend, :], axes=1)
        return res


class Fourier(ParametricFunction):
    """
    The Fourier series f(x) = c x
    """

    def __init__(self, order=1, freq=1.0, start_order=0, domain=None, output_shape=(), coefficients=None):
        if domain is None:
            domain = Domain.Td(dim=1)
        self.order = 2 * order + 1
        self.start_order = start_order
        self.freq = freq
        self.n_functions_features_ = domain.dim * (self.order - self.start_order)
        super().__init__(domain, output_shape, coefficients)

    def transform(self, x, *args, **kwargs):
        _, dim = x.shape
        res = 0.0
        for n in range(self.start_order, self.order):
            istart = (n - self.start_order) * dim
            iend = (n + 1 - self.start_order) * dim
            if n == 0:
                res += np.ones_like(x) / np.sqrt(2 * np.pi) @ self._coefficients[istart:iend, :]
            elif n % 2 == 0:
                # print(n / 2)
                res += np.cos(n / 2 * x * self.freq) / np.sqrt(np.pi) @ self._coefficients[istart:iend, :]
            else:
                # print((n + 1) / 2)
                res += np.sin((n + 1) / 2 * x * self.freq) / np.sqrt(np.pi) @ self._coefficients[istart:iend, :]
        return res

    def transform_dx(self, x, *args, **kwargs):
        _, dim = x.shape
        res = 0.0
        for n in range(self.start_order, self.order):  # First value is zero anyway
            istart = n * dim
            iend = (n + 1 - self.start_order) * dim

            if n % 2 == 0:
                grad = (-n / 2 * self.freq * np.sin(n / 2 * self.freq * x) / np.sqrt(np.pi))[..., None] * np.eye(dim)[None, :, :]
            else:
                grad = ((n + 1) / 2 * self.freq * np.cos((n + 1) / 2 * self.freq * x) / np.sqrt(np.pi))[..., None] * np.eye(dim)[None, :, :]
            res += np.einsum("nbd,bs->nsd", grad, self._coefficients[istart:iend, :])  # .reshape(-1, *self.output_shape_, dim)
        return res

    def transform_dcoeffs(self, x, *args, **kwargs):
        _, dim = x.shape
        transform_dcoeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        res = 0.0
        for n in range(self.start_order, self.order):
            istart = (n - self.start_order) * dim
            iend = (n + 1 - self.start_order) * dim
            if n == 0:
                basis = np.ones_like(x) / np.sqrt(2 * np.pi)
            elif n % 2 == 0:
                basis = np.cos(n / 2 * x * self.freq) / np.sqrt(np.pi)
            else:
                basis = np.sin((n + 1) / 2 * x * self.freq) / np.sqrt(np.pi)

            res += np.tensordot(basis, transform_dcoeffs[istart:iend, :], axes=1)
        return res
