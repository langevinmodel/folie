from .base import ParametricFunction
import numpy as np
from scipy.interpolate import make_interp_spline, make_lsq_spline, BSpline
from ..data import stats_from_input_data
from sklearn.preprocessing import SplineTransformer


class BSplinesFunction(ParametricFunction):
    """
    A function that use a set of B-splines
    """

    def __init__(self, knots=5, k=3, bc_type=None, output_shape=(), coefficients=None):
        super().__init__(output_shape, coefficients)
        self.bc_type = bc_type
        self.k = k
        self.knots = knots
        self.input_dim_ = 1
        self.bspline = make_interp_spline(np.linspace(0.0, 1.0, 15), np.zeros(15), k=self.k, bc_type=self.bc_type)  # Defaut bspline for the sake of initialisation

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

    def fit(self, X, y=None, **kwargs):
        xstats = stats_from_input_data(X[:, : self.dim_x])
        t = self._build_knots_array(xstats)
        self.input_dim_ = xstats.dim
        x, unique_indices = np.unique(X[:, 0], return_index=True)
        if y is None:
            y = np.zeros((x.shape[0], self.input_dim_, self.output_size_))
        elif y.shape[0] != xstats.nobs:
            raise ValueError("y should be of the same length than X ")
        else:
            y = y[unique_indices, ...].reshape((x.shape[0], self.input_dim_, self.output_size_))
        self.bspline = make_lsq_spline(x, y, t, k=self.k)  # Get more efficient algo that does not assume to have a set of  points representing a curve
        self.n_functions_features_ = self.bspline.c.shape[0]
        self.fitted_ = True
        return self

    def differentiate(self):
        # TODO: Implement the differentiation
        fun = BSplinesFunction(np.r_[self.output_shape_, [self.input_dim_]])
        self.output_shape_ = np.asarray(output_shape, dtype=int)
        fun = self.copy()
        # fun.bspline=

    def resize(self, new_shape):
        super().resize(new_shape)
        self.bspline.c = np.resize(self.bspline.c, (self.n_functions_features_, self.input_dim_, self.output_size_))
        return self

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

    def transform(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        return np.trace(self.bspline(x), axis1=1, axis2=2)

    def transform_x(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        return np.diagonal(self.bspline.derivative()(x), axis1=1, axis2=2).reshape(nsamples, *self.output_shape_, dim)

    def hessian_x(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        return np.diagonal(self.bspline.derivative(2)(x), axis1=1, axis2=2).reshape(nsamples, *self.output_shape_, dim, dim)

    def transform_coeffs(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, self.input_dim_, *self.output_shape_, self.size)
        return np.trace(BSpline(self.bspline.t, transform_coeffs, self.bspline.k)(x), axis1=1, axis2=2)

        # Aternative via design matrix
        BSpline.design_matrix(x, self.bspline.t, self.bspline.k)  # return (Nobs x nelemnts_basis) en format sparse CSR


# En vrai, ci dessous ça utilise aussi Bspline, copier le code relevant pour merger les 2
class sklearnBSplines(ParametricFunction):
    """
    A slower but more complete set of BSplines
    """

    def __init__(self, knots=5, k=3, bc_type="continue", output_shape=(), coefficients=None):
        super().__init__(output_shape, coefficients)
        self.bc_type = bc_type
        self.k = k
        self.knots = knots

    def fit(self, X, y=None, **kwargs):
        self.bspline = SplineTransformer(n_knots=self.knots, degree=self.k, extrapolation=self.bc_type, sparse_output=True).fit(X[:, : self.dim_x])
        self.n_functions_features_ = self.bspline.n_features_out_
        super().fit(X[:, : self.dim_x], y, **kwargs)
        return self

    def transform(self, x, *args, **kwargs):
        return self.bspline.transform(x) @ self._coefficients

    def transform_x(self, x, *args, **kwargs):
        len, dim = x.shape
        x_grad = np.ones((len, 1, 1)) * np.eye(dim)[None, :, :]  # TODO: a corriger
        return np.einsum("nbd,bs->nsd", x_grad, self._coefficients)  # .reshape(-1, *self.output_shape_, dim)

    def transform_coeffs(self, x, *args, **kwargs):
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        return np.tensordot(self.bspline.transform(x), transform_coeffs, axes=1)
