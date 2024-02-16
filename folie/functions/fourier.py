import numpy as np
from .base import FunctionFromBasis
from ..data import stats_from_input_data


class Fourier(FunctionFromBasis):
    """
    The linear function f(x) = c x
    """

    def __init__(self, order=1, freq=1.0, output_shape=()):
        super().__init__(output_shape)
        self.order = 2 * order + 1
        self.freq = freq

    def fit(self, X=None, y=None):
        xstats = stats_from_input_data(X)
        self.input_dim_ = xstats.dim
        self.n_basis_features_ = self.input_dim_ * self.order
        self.coefficients = np.ones((self.n_basis_features_, self.output_size_))
        return self

    def differentiate(self):
        fun = Fourier(self.output_shape)
        fun.coefficients = self.coefficients.copy()
        return fun

    def transform(self, x, **kwargs):
        nsamples, dim = x.shape
        features = np.zeros((nsamples, self.output_size_))
        for n in range(0, self.order):
            istart = n * dim
            iend = (n + 1) * dim
            if n == 0:
                features[:, istart:iend] = np.ones_like(x) / np.sqrt(2 * np.pi)
            elif n % 2 == 0:
                # print(n / 2)
                features[:, istart:iend] = np.cos(n / 2 * x * self.freq) / np.sqrt(np.pi)
            else:
                # print((n + 1) / 2)
                features[:, istart:iend] = np.sin((n + 1) / 2 * x * self.freq) / np.sqrt(np.pi)
        return features

    def grad_x(self, x, **kwargs):
        len, dim = x.shape
        x_grad = np.ones((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients).reshape(-1, *self.output_shape_, dim)

    def hessian_x(self, x, **kwargs):
        len, dim = x.shape
        x_grad = np.ones((len, 1, 1)) * np.eye(dim)[None, :, :]
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients).reshape(-1, *self.output_shape_, dim)

    def grad_coeffs(self, x, **kwargs):
        grad_coeffs = np.eye(self.size).reshape(self.n_basis_features_, *self.output_shape_, self.size)
        return np.tensordot(x, grad_coeffs, axes=1)
