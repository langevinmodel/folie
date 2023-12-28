from .base import FunctionFromBasis
import numpy as np
import skfem


class FiniteElementFunction(FunctionFromBasis):
    """
    Build functions from finite elements basis
    """

    def __init__(self, basis, output_shape=()):
        super().__init__(output_shape)
        self.basis = basis

    def fit(self, x, y=None):
        _, dim = x.shape
        self.n_basis_features_ = self.basis.N
        self.coefficients = np.zeros((self.n_basis_features_, self.output_size_))
        return self

    def transform(self, x, **kwargs):
        return self.basis.probes(x) @ self._coefficients

    def grad_x(self, x, **kwargs):
        _, dim = x.shape
        # Reprendre le code de basis.probes et l'adapter pour avoir la dérivée
        return np.einsum("nbd,bs->nsd", np.ones_like(x), self._coefficients).reshape(-1, *self.output_shape_, dim)

    def grad_coeffs(self, x, **kwargs):
        #  Changer ke ones_like pour avoir le gradient
        grad_coeffs = (np.ones((self.n_basis_features_, 1, 1)) * np.eye(self.output_size_)[None, :, :]).reshape(-1, *self.output_shape_, self.size)
        return self.basis.probes(x) @ grad_coeffs
