from .._numpy import np
from .base import ParametricFunction
from ..domains import Domain


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

    def differentiate(self):
        fun = Fourier(self.output_shape)  # Use start_order to remove first value
        fun.coefficients = self.coefficients.copy()
        return fun

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

    def transform_x(self, x, *args, **kwargs):
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

    def transform_coeffs(self, x, *args, **kwargs):
        _, dim = x.shape
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
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

            res += np.tensordot(basis, transform_coeffs[istart:iend, :], axes=1)
        return res