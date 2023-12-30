from .base import FunctionFromBasis
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from ..data import Trajectories, traj_stats


class BSplinesFunction(FunctionFromBasis):
    """
    A function that use a set of B-splines
    """

    def __init__(self, output_shape=(), knots=5, k=3, bc_type=None):
        super().__init__(output_shape)
        self.bc_type = bc_type
        self.k = k
        self.knots = knots

    def fit(self, x, y=None):
        if isinstance(x, Trajectories):
            xstats = x.stats
        else:
            xstats = traj_stats(x)
        dim = xstats.dim
        if isinstance(self.knots, int):
            self.x_spline = np.linspace(xstats.min[0], xstats.max[0], self.knots)
        else:
            self.x_spline = np.unique(self.knots)
        self.input_dim_ = dim
        nknots = len(self.x_spline)
        if y is None:
            y = np.zeros((nknots, self.input_dim_, self.output_size_))
        elif y.shape[0] != nknots:
            raise ValueError("y should be of length of the number of knots ")
        self.bspline = make_interp_spline(self.x_spline, y, k=self.k, bc_type=self.bc_type)
        self.n_basis_features_ = self.bspline.c.shape[0]
        return self

    def resize(self, new_shape):
        super().resize(new_shape)
        self.bspline.c = np.resize(self.bspline.c, (self.n_basis_features_, self.input_dim_, self.output_size_))
        return self

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self.bspline.c.ravel()

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.bspline.c = vals.reshape((self.n_basis_features_, self.input_dim_, self.output_size_))

    @property
    def size(self):
        return np.size(self.bspline.c)

    def transform(self, x, **kwargs):
        nsamples, dim = x.shape
        return np.trace(self.bspline(x), axis1=1, axis2=2)

    def grad_x(self, x, **kwargs):
        nsamples, dim = x.shape
        return np.diagonal(self.bspline.derivative()(x), axis1=1, axis2=2).reshape(nsamples, *self.output_shape_, dim)

    def grad_coeffs(self, x, **kwargs):
        nsamples, dim = x.shape
        grad_coeffs = np.eye(self.size).reshape(self.n_basis_features_, self.input_dim_, *self.output_shape_, self.size)
        print(grad_coeffs.shape)
        return np.trace(BSpline(self.bspline.t, grad_coeffs, self.bspline.k)(x), axis1=1, axis2=2)
