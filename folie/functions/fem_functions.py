from .base import ParametricFunction
from .._numpy import np
import sparse
import skfem


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
        return probes_matrix.tocsr() @ self._coefficients  # Check efficiency of conversion to csr

    def transform_x(self, x, *args, **kwargs):
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

    def transform_coeffs(self, x, *args, **kwargs):
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
            transform_coeffs = np.eye(self.size).reshape((self.n_functions_features_, self.output_size_, self.size))
            return sparse.einsum("nb,bsc->nsc", probes_matrix, transform_coeffs)
