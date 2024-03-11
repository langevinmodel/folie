from .base import ParametricFunction
from .._numpy import np
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.preprocessing import SplineTransformer
import sparse


class BSplinesFunction(ParametricFunction):
    """
    A function that use a set of B-splines. This is an overlay above scipy BSpline object that hold coefficients and most parameters
    """

    def __init__(self, domain, k=3, bc_type=None, output_shape=(), coefficients=None):
        if domain.dim > 1:
            raise ValueError("BSplinesFunction does not handle higher dimensionnal system")
        knots = domain.mesh.p.squeeze()
        self.bspline = make_interp_spline(knots, np.zeros(len(knots)), k=k, bc_type=bc_type)
        self.n_functions_features_ = self.bspline.c.shape[0]
        super().__init__(domain, output_shape, coefficients)

    def differentiate(self):
        # TODO: Implement the differentiation
        fun = BSplinesFunction(np.r_[self.output_shape_, [self.input_dim_]])
        self.output_shape_ = np.asarray(output_shape, dtype=int)
        fun = self.copy()
        # fun.bspline=

    def resize(self, new_shape):
        super().resize(new_shape)
        self.bspline.c = np.resize(self.bspline.c, (self.n_functions_features_, self.output_size_))
        return self

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self.bspline.c.ravel()

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.bspline.c = vals.reshape((self.n_functions_features_, self.output_size_))

    @property
    def size(self):
        return np.size(self.bspline.c)

    def transform(self, x, *args, **kwargs):
        return self.bspline(x)

    def transform_x(self, x, *args, **kwargs):
        return self.bspline.derivative()(x).reshape(x.shape[0], self.output_size_, x.shape[-1])

    def transform_xx(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        return self.bspline.derivative(2)(x).reshape(nsamples, *self.output_shape_, dim, dim)

    def transform_coeffs(self, x, *args, **kwargs):
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        return np.trace(BSpline(self.bspline.t, transform_coeffs, self.bspline.k)(x), axis1=1, axis2=2)
        # transform_coeffs = sparse.COO.from_numpy(np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size))
        # dmat = sparse.COO.from_scipy_sparse(BSpline.design_matrix(x[:, 0], self.bspline.t, self.bspline.k))  # return (Nobs x nelemnts_basis) en format sparse CSR that we convert into sparse matrix
        # return sparse.einsum("nb,bsc->nsc", dmat, transform_coeffs)


# En vrai, ci dessous ça utilise aussi Bspline, copier le code relevant pour merger les 2
class sklearnBSplines(ParametricFunction):
    """
    A slower but more complete set of BSplines
    """

    def __init__(self, domain, k=3, bc_type="continue", output_shape=(), coefficients=None):

        assert domain.dim <= 1
        self.bc_type = bc_type
        self.k = k
        self.knots = domain.mesh.p.T
        # Si les knots sont sont en mode quantile, mais en vrai on va le faire en externe donc knots est toujours un array
        self.bspline = SplineTransformer(n_knots=self.knots.shape[0], degree=self.k, extrapolation=self.bc_type, sparse_output=True).fit(self.knots)
        self.n_functions_features_ = self.bspline.n_features_out_
        super().__init__(domain, output_shape, coefficients)

    def transform(self, x, *args, **kwargs):
        return self.bspline.transform(x) @ self._coefficients

    def transform_coeffs(self, x, *args, **kwargs):
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, -1)
        return self.bspline.transform(x) @ transform_coeffs
