from .base import FunctionFromBasis
import numpy as np
from scipy.interpolate import splev, splrep


class BSplinesFunction(FunctionFromBasis):
    """
    A function that use a set of B-splines
    """

    def __init__(self, output_shape=(), knots=5, k=3, periodic=False):
        super().__init__(output_shape)
        self.periodic = periodic
        self.k = k
        self.knots = knots
        self.extra_zeros = np.zeros(self.k + 1)

    def fit(self, x, y=None):
        _, dim = x.shape
        if isinstance(self.knots, int):
            x_range = np.linspace(np.min(x), np.max(x), self.knots)
        else:
            x_range = self.knots

        nknots = len(x_range)
        if y is None:
            y = np.zeros(nknots)
        elif y.shape[0] != nknots:
            raise ValueError("y should be of length of the number of knots ")
        knots, c, _ = splrep(x_range, y, k=self.k, per=self.periodic)
        self.knots_ = knots
        self._coefficients = c[: -(self.k + 1)]
        return self

    def transform(self, x, **kwargs):
        nsamples, dim = x.shape
        res = np.empty(nsamples, *self.output_shape_)
        return splev(x, (self.knots_, np.concatenate((self.coefficients, self.extra_zeros)), self.k))

    def grad_x(self, x, **kwargs):
        return splev(x, (self.knots_, np.concatenate((self.coefficients, self.extra_zeros)), self.k), der=1)

    def grad_coeffs(self, x, **kwargs):
        nsamples, dim = x.shape
        features = np.zeros((nsamples, dim * self.coefficients.shape[0]))
        ncoeffs = len(self.knots_)
        for ispline in range(self.coefficients.shape[0]):
            istart = ispline * dim
            iend = (ispline + 1) * dim
            coeffs = np.asarray([1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)])
            features[:, istart:iend] = splev(x, (self.knots_, coeffs, self.k))
        return features

    # def zero(self):
    #     r"""Get the coefficients to evaluate the function to zero."""
    #     self._coefficients = np.zeros((len(self.knots_) - self.k - 1,) + self.output_shape_)
    #     return self

    # def one(self):
    #     r"""Get the coefficients to evaluate the function to one"""
    #     self._coefficients = np.ones((len(self.knots_) - self.k - 1,) + self.output_shape_)
    #     return self

