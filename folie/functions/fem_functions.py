from .base import ParametricFunction
import numpy as np
from ..data import stats_from_input_data


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
        cells = self.basis.mesh.element_finder(mapping=self.mapping)(*x)  # Change the element finder
        # Find a way to exclude out of mesh elements, we can define an outside elements that is a constant
        pts = self.mapping.invF(x[:, :, np.newaxis], tind=cells)
        phis = np.array([self.elem.gbasis(self.mapping, pts, k, tind=cells)[0] for k in range(self.Nbfun)]).flatten()
        probes_matrix = coo_matrix(
            (
                phis,
                (
                    np.tile(np.arange(x.shape[1]), self.Nbfun),
                    self.element_dofs[:, cells].flatten(),
                ),
            ),
            shape=(x.shape[0], self.N),
        )  # Use sparse array

        return probes_matrix @ self._coefficients

    def transform_x(self, x, *args, **kwargs):
        _, dim = x.shape
        cells = self.basis.mesh.element_finder(mapping=self.mapping)(*x)  # Change the element finder
        # Find a way to exclude out of mesh elements, we can define an outside elements that is a constant
        pts = self.mapping.invF(x[:, :, np.newaxis], tind=cells)
        phis = np.array([self.elem.gbasis(self.mapping, pts, k, tind=cells)[0] for k in range(self.Nbfun)]).flatten()
        return coo_matrix(
            (
                phis,
                (
                    np.tile(np.arange(x.shape[1]), self.Nbfun),
                    self.element_dofs[:, cells].flatten(),
                ),
            ),
            shape=(x.shape[0], self.N),
        )
        # Its return a matrix, we should either trying to do the sum, or use sparse array from the sparse library

        # Reprendre le code de basis.probes et l'adapter pour avoir la dérivée
        return np.einsum("nbd,bs->nsd", np.ones_like(x), self._coefficients).reshape(-1, *self.output_shape_, dim)

    def transform_coeffs(self, x, *args, **kwargs):
        # TODO: Retourne une matrice sparse
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, *self.output_shape_, self.size)
        return np.tensordot(self.basis.probes(x), transform_coeffs, axes=1)
