import numpy as np
from . import Basis

import scipy.interpolate


class SplineFct(Basis):
    """
    A single basis function that is given from splines fit of data
    """

    def __init__(self, knots, coeffs, k=3, periodic=False):
        self.periodic = periodic
        self.k = k
        self.t = knots  # knots are position along the axis of the knots
        self.c = coeffs
        self.const_removed = False
        self.dim_out_basis = 1

    def fit(self, X):
        self.spl_ = scipy.interpolate.BSpline(self.t, self.c, self.k)
        self.n_output_features_ = X.dim
        return self

    def transform(self, X, **kwargs):
        return self.spl_(X)

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        grad = np.zeros((nsamples, dim) + (dim,) * deriv_order)
        for i in range(dim):
            grad[(Ellipsis, slice(i, i + 1)) + (i,) * (deriv_order)] = self.spl_.derivative(deriv_order)(X[:, slice(i, i + 1)])
        return grad

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        return self.spl_.antiderivative(order)(X)


def _get_bspline_basis(knots, degree=3, periodic=False):
    """Get spline coefficients for each basis spline."""
    nknots = len(knots)
    y_dummy = np.zeros(nknots)

    knots, coeffs, degree = scipy.interpolate.splrep(knots, y_dummy, k=degree, per=periodic)
    ncoeffs = len(coeffs)
    bsplines = []
    for ispline in range(nknots):
        coeffs = np.asarray([1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)])
        bsplines.append((knots, coeffs, degree))
    return bsplines


class BSplines(Basis):  # TODO replace current implementation by one using Bspline.basis_element
    """
    Bsplines features class
    """

    def __init__(self, n_knots=5, k=3, periodic=False, remove_const=True):
        """
        Parameters
        ----------
        n_knots : int
            Number of knots to use
        k : int
            Degree of the splines
        periodic: bool
            Whatever to use periodic splines or not
        """
        self.periodic = periodic
        self.k = k
        self.n_knots = n_knots  # knots are position along the axis of the knots
        self.const_removed = remove_const
        self.dim_out_basis = 1

    def fit(self, X, knots=None):
        dim = X.dim
        # TODO determine non uniform position of knots given the datas
        if knots is None:
            knots = np.linspace(X.stats.min, X.stats.max, self.n_knots)
        self.bsplines_ = _get_bspline_basis(knots, self.k, periodic=self.periodic)
        self._nsplines = len(self.bsplines_)
        self.n_output_features_ = len(self.bsplines_) * dim
        return self

    def transform(self, X):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * dim
            iend = (ispline + 1) * dim
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self._nsplines - with_const)) + (dim,) * deriv_order)
        if self.k < deriv_order:
            return features
        for ispline, spline in enumerate(self.bsplines_[: len(self.bsplines_) - with_const]):
            istart = (ispline) * dim
            for i in range(dim):
                features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = scipy.interpolate.splev(X[:, slice(i, i + 1)], spline, der=deriv_order)
        return features

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * dim
            iend = (ispline + 1) * dim
            spline_int = scipy.interpolate.splantider(spline, n=order)
            features[:, istart:iend] = scipy.interpolate.splev(X, spline_int)
        return features
