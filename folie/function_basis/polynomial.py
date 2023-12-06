import numpy as np
from . import Basis


class Linear(Basis):
    """
    Linear function
    """

    def __init__(self, to_center=False):
        """"""
        self.centered = to_center
        self.const_removed = False

    def fit(self, X, y=None):
        self.n_output_features_ = X.dim
        self.dim_out_basis = 1
        if self.centered:
            self.mean_ = X.stats.mean
        else:
            self.mean_ = np.zeros((self.n_output_features_,))
        return self

    def transform(self, X, **kwargs):
        return X - self.mean_

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        grad = np.zeros((nsamples, dim) + (dim,) * deriv_order)
        if deriv_order == 1:
            for i in range(dim):
                grad[..., i, i] = 1.0
        return grad

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        return 0.5 * np.power(X, 2)


class Polynomial(Basis):
    """
    Wrapper for numpy polynomial series.
    """

    def __init__(self, deg=1, polynom=np.polynomial.Polynomial, remove_const=True):
        """
        Providing a numpy polynomial class via polynom keyword allow to change polynomial type.
        """
        self.degree = deg + 1
        self.polynom = polynom
        self.const_removed = remove_const

    def fit(self, X, y=None):
        self.n_output_features_ = X.dim * self.degree
        self.dim_out_basis = 1
        return self

    def transform(self, X, **kwargs):
        nsamples, dim = X.shape

        features = np.zeros((nsamples, dim * self.degree))
        for n in range(0, self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            features[:, istart:iend] = self.polynom.basis(n)(X)
        # print("Basis", X.shape, features.shape)
        return features

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self.degree - with_const)) + (dim,) * deriv_order)
        for n in range(with_const, self.degree):
            istart = (n - with_const) * dim
            for i in range(dim):
                features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = self.polynom.basis(n).deriv(deriv_order)(X[:, slice(i, i + 1)])
        return features

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, dim * self.degree))
        for n in range(0, self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            features[:, istart:iend] = self.polynom.basis(n).integ(order)(X)
        return features
