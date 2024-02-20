from .base import ParametricFunction
import numpy as np
from scipy.interpolate import make_interp_spline, make_lsq_spline, BSpline
from ..data import stats_from_input_data


class BSplinesFunction(ParametricFunction):
    """
    A function that use a set of B-splines
    """

    def __init__(self, knots=5, k=3, bc_type=None, output_shape=(), coefficients=None):
        super().__init__(output_shape, coefficients)
        self.bc_type = bc_type
        self.k = k
        self.knots = knots

    def _build_knots_array(self, xstats):
        """
        Helper function to build the knots array
        """
        if isinstance(self.knots, int):
            X = np.linspace(xstats.min[0], xstats.max[0], self.knots)
        else:
            X = np.unique(self.knots.ravel())
        bspl = make_interp_spline(X, np.zeros(len(X)), k=self.k, bc_type=self.bc_type)
        return bspl.t

    def fit(self, X, y=None):
        xstats = stats_from_input_data(X)
        t = self._build_knots_array(xstats)
        self.input_dim_ = xstats.dim
        if y is None:
            y = np.zeros((xstats.nobs, self.input_dim_, self.output_size_))
        elif y.shape[0] != xstats.nobs:
            raise ValueError("y should be of the same length than X ")
        else:
            y = y.reshape((xstats.nobs, self.input_dim_, self.output_size_))
        self.bspline = make_lsq_spline(X[:, 0], y, t, k=self.k)
        self.n_functions_features_ = self.bspline.c.shape[0]
        return self

    def differentiate(self):
        fun = BSplinesFunction(np.r_[self.output_shape_, [self.input_dim_]])
        self.output_shape_ = np.asarray(output_shape, dtype=int)
        fun = self.copy()
        # fun.bspline=

    def resize(self, new_shape):
        super().resize(new_shape)
        self.bspline.c = np.resize(self.bspline.c, (self.n_functions_features_, self.input_dim_, self.output_size_))
        return self

    @property
    def size(self):
        return np.size(self._coefficients)

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self.bspline.c.ravel()

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.bspline.c = vals.reshape((self.n_functions_features_, self.input_dim_, self.output_size_))

    @property
    def size(self):
        return np.size(self.bspline.c)

    def transform(self, x, **kwargs):
        nsamples, dim = x.shape
        return np.trace(self.bspline(x), axis1=1, axis2=2)

    def grad_x(self, x, **kwargs):
        nsamples, dim = x.shape
        return np.diagonal(self.bspline.derivative()(x), axis1=1, axis2=2).reshape(nsamples, *self.output_shape_, dim)

    def hessian_x(self, x, **kwargs):
        nsamples, dim = x.shape
        return np.diagonal(self.bspline.derivative(2)(x), axis1=1, axis2=2).reshape(nsamples, *self.output_shape_, dim, dim)

    def grad_coeffs(self, x, **kwargs):
        nsamples, dim = x.shape
        grad_coeffs = np.eye(self.size).reshape(self.n_functions_features_, self.input_dim_, *self.output_shape_, self.size)
        return np.trace(BSpline(self.bspline.t, grad_coeffs, self.bspline.k)(x), axis1=1, axis2=2)
