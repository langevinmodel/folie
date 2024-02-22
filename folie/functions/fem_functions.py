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

    def fit(self, X=None, y=None, **kwargs):
        xstats = stats_from_input_data(X)
        xstats.dim
        self.n_functions_features_ = self.basis.N
        self.coefficients = np.zeros((self.n_functions_features_, self.output_size_))
        return self

    def transform(self, x, *args, **kwargs):
        return self.basis.probes(x) @ self._coefficients

    def transform_x(self, x, *args, **kwargs):
        _, dim = x.shape
        # Reprendre le code de basis.probes et l'adapter pour avoir la dérivée
        return np.einsum("nbd,bs->nsd", np.ones_like(x), self._coefficients).reshape(-1, *self.output_shape_, dim)

    def transform_coeffs(self, x, *args, **kwargs):
        # TODO: Retourne une matrice sparse
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, *self.output_shape_, self.size)
        return np.tensordot(self.basis.probes(x), transform_coeffs, axes=1)
