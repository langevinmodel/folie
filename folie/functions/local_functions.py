from .base import ParametricFunction
from .._numpy import np
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.preprocessing import SplineTransformer
import sparse
import skfem
import numba as nb


class FiniteElement(ParametricFunction):
    """
    Build functions from finite elements basis
    """

    def __init__(self, domain, element, output_shape=(), coefficients=None):
        self.basis = skfem.CellBasis(domain.mesh, element)
        self.n_functions_features_ = self.basis.N
        super().__init__(domain, output_shape, coefficients)

    def transform(self, x, *args, **kwargs):
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.domain.localize_data(x)

        res = 0.0
        for k in range(self.basis.Nbfun):
            res += self.basis.elem.gbasis(self.basis.mapping, loc_x.T[..., None], k, tind=cells)[0] * self._coefficients[self.basis.element_dofs[k, cells], ...]
        return res

    def transform_dx(self, x, *args, **kwargs):
        _, dim = x.shape
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.domain.localize_data(x)
        phis = np.array([self.basis.elem.gbasis(self.basis.mapping, loc_x.T[..., None], k, tind=cells)[0].grad for k in range(self.basis.Nbfun)]).ravel()
        probes_matrix = sparse.COO(
            np.array(
                [
                    np.tile(np.arange(x.shape[0]), self.basis.Nbfun * dim),
                    np.repeat(self.basis.element_dofs[:, cells].flatten(), dim),  # Il faut donc pad les elements dofs avec numpy tile et rajouter une troisème colonne
                    np.tile(np.repeat(np.arange(dim), x.shape[0]), self.basis.Nbfun),
                ]
            ),
            phis,
            shape=(x.shape[0], self.basis.N, dim),
        )
        return sparse.einsum("nbd,bs->nsd", probes_matrix, self._coefficients)

    def transform_dcoeffs(self, x, *args, **kwargs):
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.domain.localize_data(x)
        phis = np.array([self.basis.elem.gbasis(self.basis.mapping, loc_x.T[..., None], k, tind=cells)[0] for k in range(self.basis.Nbfun)]).flatten()
        probes_matrix = sparse.COO(
            np.array(
                [
                    np.tile(np.arange(x.shape[0]), self.basis.Nbfun),
                    self.basis.element_dofs[:, cells].flatten(),
                ]
            ),
            phis,
            shape=(x.shape[0], self.basis.N),
        )
        if self.size == self.basis.N:  # C'est peut-être une autre condition plus générale, genre un gcd?
            return probes_matrix.reshape((-1, self.output_size_, self.basis.N))
        else:
            transform_dcoeffs = sparse.COO.from_numpy(np.eye(self.size).reshape((self.n_functions_features_, self.output_size_, self.size)))
            return sparse.einsum("nb,bsc->nsc", probes_matrix, transform_dcoeffs)


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

    def transform_dx(self, x, *args, **kwargs):
        return self.bspline.derivative()(x).reshape(x.shape[0], self.output_size_, x.shape[-1])

    def transform_d2x(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        return self.bspline.derivative(2)(x).reshape(nsamples, *self.output_shape_, dim, dim)

    def transform_dcoeffs(self, x, *args, **kwargs):
        transform_dcoeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        return np.trace(BSpline(self.bspline.t, transform_dcoeffs, self.bspline.k)(x), axis1=1, axis2=2)
        # transform_dcoeffs = sparse.COO.from_numpy(np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size))
        # dmat = sparse.COO.from_scipy_sparse(BSpline.design_matrix(x[:, 0], self.bspline.t, self.bspline.k))  # return (Nobs x nelemnts_basis) en format sparse CSR that we convert into sparse matrix
        # return sparse.einsum("nb,bsc->nsc", dmat, transform_dcoeffs)


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

    def transform_dcoeffs(self, x, *args, **kwargs):
        transform_dcoeffs = np.eye(self.size).reshape(self.n_functions_features_, -1)
        return self.bspline.transform(x) @ transform_dcoeffs


@nb.njit
def linear_interpolation(idx, h, fp):  # pragma: no cover
    f0, f1 = fp[idx - 1], fp[idx]
    hm = 1 - h
    val = hm * f0 + h * f1
    return val


@nb.njit
def linear_interpolation_x(idx, h, fp):  # pragma: no cover
    f0, f1 = fp[idx - 1], fp[idx]
    return f0 - f1


@nb.njit
def linear_element_gradient(idx, h, n_coeffs):  # pragma: no cover
    hm = 1 - h
    # Set gradient elements one by one
    grad = np.zeros((n_coeffs, idx.shape[0]))
    for i, ik in enumerate(idx):
        grad[ik - 1, i] = hm[i]
        grad[ik, i] = h[i]
    return grad


class Optimized1DLinearElement(ParametricFunction):
    def __init__(self, domain, element, output_shape=(), coefficients=None):
        self.basis = skfem.CellBasis(domain.mesh, element)
        self.n_functions_features_ = self.basis.N
        super().__init__(domain, output_shape, coefficients)

    def transform(self, x, *args, **kwargs):
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.domain.localize_data(x)
        return linear_interpolation(cells, loc_x, self.coefficients)  # Check shape

    def transform_dx(self, x, *args, **kwargs):
        _, dim = x.shape
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.domain.localize_data(x)
        return linear_interpolation_x(cells, loc_x, self.coefficients)

    def transform_dcoeffs(self, x, *args, **kwargs):
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.domain.localize_data(x)
        return linear_element_gradient(cells, loc_x, self.size)
