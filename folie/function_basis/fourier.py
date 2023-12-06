import numpy as np
from . import Basis


class Fourier(Basis):
    """
    Fourier series.
    """

    def __init__(self, order=1, freq=1.0, remove_const=True):
        """
        Parameters
        ----------
        order :  int
            Order of the Fourier series
        freq: float
            Base frequency
        """
        self.order = 2 * order + 1
        self.freq = freq
        self.const_removed = remove_const

    def fit(self, X):
        self.n_output_features_ = X.dim * self.order
        self.dim_out_basis = 1
        return self

    def transform(self, X, **kwargs):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, dim * self.order))
        for n in range(0, self.order):
            istart = n * dim
            iend = (n + 1) * dim
            if n == 0:
                features[:, istart:iend] = np.ones_like(X) / np.sqrt(2 * np.pi)
            elif n % 2 == 0:
                # print(n / 2)
                features[:, istart:iend] = np.cos(n / 2 * X * self.freq) / np.sqrt(np.pi)
            else:
                # print((n + 1) / 2)
                features[:, istart:iend] = np.sin((n + 1) / 2 * X * self.freq) / np.sqrt(np.pi)
        return features

    def deriv(self, X, deriv_order=1):
        if deriv_order == 2:
            return self.hessian(X)
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self.order - with_const)) + (dim,) * deriv_order)
        for n in range(with_const, self.order):
            istart = (n - with_const) * dim
            for i in range(dim):
                if n % 2 == 0:
                    features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = -n / 2 * self.freq * np.sin(n / 2 * self.freq * X[:, slice(i, i + 1)]) / np.sqrt(np.pi)
                else:
                    features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = (n + 1) / 2 * self.freq * np.cos((n + 1) / 2 * self.freq * X[:, slice(i, i + 1)]) / np.sqrt(np.pi)
        return features

    def hessian(self, X):
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self.order - with_const)) + (dim,) * 2)
        for n in range(with_const, self.order):
            istart = (n - with_const) * dim
            for i in range(dim):
                if n % 2 == 0:
                    features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * 2] = -(((n / 2) * self.freq) ** 2) * np.cos(n / 2 * self.freq * X[:, slice(i, i + 1)]) / np.sqrt(np.pi)
                else:
                    features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * 2] = -(((n + 1) / 2 * self.freq) ** 2) * np.sin((n + 1) / 2 * self.freq * X[:, slice(i, i + 1)]) / np.sqrt(np.pi)
        return features

    def antiderivative(self, X, order=1):
        if order > 1:
            raise NotImplementedError
        nsamples, dim = X.shape
        features = np.zeros((nsamples, dim * self.order))
        for n in range(0, self.order):
            istart = n * dim
            iend = (n + 1) * dim
            if n == 0:
                features[:, istart:iend] = X / np.sqrt(2 * np.pi)
            elif n % 2 == 0:
                # print(n / 2)
                features[:, istart:iend] = np.sin(n / 2 * X * self.freq) / (np.sqrt(np.pi) * n / 2 * self.freq)
            else:
                # print((n + 1) / 2)
                features[:, istart:iend] = -1 * np.cos((n + 1) / 2 * X * self.freq) / (np.sqrt(np.pi) * (n + 1) / 2 * self.freq)
        return features
