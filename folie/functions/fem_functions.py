from .base import ParametricFunction
import numpy as np
from ..data import stats_from_input_data
import sparse


class FiniteElement(ParametricFunction):
    """
    Build functions from finite elements basis
    """

    def __init__(self, basis, output_shape=(), coefficients=None):
        super().__init__(output_shape, coefficients)
        self.basis = basis
        self.n_functions_features_ = self.basis.N

    def preprocess_traj(self, X):
        """
        Get elements and position within the elements
        """
        cells = self.basis.mesh.element_finder(mapping=self.mapping)(*(X.T))  # Change the element finder
        # Find a way to exclude out of mesh elements, we can define an outside elements that is a constant
        pts = self.mapping.invF(X.T[:, :, np.newaxis], tind=cells)
        return cells, pts

    def fit(self, X=None, y=None, **kwargs):
        xstats = stats_from_input_data(X)
        xstats.dim
        self.coefficients = np.zeros((self.n_functions_features_, self.output_size_))
        self.fitted_ = True
        return self

    def transform(self, x, *args, **kwargs):
        cells = kwargs["cells"]
        loc_x = kwargs["loc_x"]
        phis = np.array([self.basis.elem.gbasis(self.basis.mapping, loc_x, k, tind=cells)[0] for k in range(self.basis.Nbfun)]).flatten()
        probes_matrix = sparse.COO(
            (
                (
                    np.tile(np.arange(x.shape[0]), self.basis.Nbfun),
                    self.basis.element_dofs[:, cells].flatten(),
                ),
                phis,
            ),
            shape=(x.shape[0], self.basis.N),
        )  # Use sparse array

        return probes_matrix.tocsr() @ self._coefficients  # Check efficiency of conversion to csr

    def transform_x(self, x, *args, **kwargs):
        _, dim = x.shape
        cells = kwargs["cells"]
        loc_x = kwargs["loc_x"]
        phis = np.array([self.basis.elem.gbasis(self.basis.mapping, loc_x, k, tind=cells)[0].grad for k in range(self.basis.Nbfun)]).flatten()  # which part of gbasis
        probes_matrix = sparse.COO(
            (
                (
                    np.tile(np.arange(x.shape[0]), self.basis.Nbfun),
                    self.basis.element_dofs[:, cells].flatten(),  # Il faut donc pad les elements dofs avec numpy tile et rajouter une troisème colonne
                ),
                phis,
            ),
            shape=(x.shape[0], self.basis.N, dim),
        )
        # Its return a matrix, we should either trying to do the sum, or use sparse array from the sparse library
        return sparse.einsum("nbd,bs->nsd", probes_matrix, self._coefficients)

    def transform_coeffs(self, x, *args, **kwargs):
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, self.output_size_, self.size)
        phis = np.array([self.basis.elem.gbasis(self.basis.mapping, loc_x, k, tind=cells)[1] for k in range(self.basis.Nbfun)]).flatten()
        probes_matrix = sparse.COO(
            (
                (
                    np.tile(np.arange(x.shape[0]), self.basis.Nbfun),
                    self.basis.element_dofs[:, cells].flatten(),
                ),
                phis,
            ),
            shape=(x.shape[0], self.basis.N),
        )
        return sparse.einsum("nb,bsc->nsc", probes_matrix, transform_coeffs)
