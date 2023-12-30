from .base import FunctionFromBasis
import numpy as np
from ..data import Trajectories, traj_stats


class FiniteElementFunction(FunctionFromBasis):
    """
    Build functions from finite elements basis
    """

    def __init__(self, basis, output_shape=()):
        super().__init__(output_shape)
        self.basis = basis

    def fit(self, x, y=None):
        if isinstance(x, Trajectories):
            xstats = x.stats
        else:
            xstats = traj_stats(x)
        dim = xstats.dim
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
        grad_coeffs = np.eye(self.size).reshape(self.n_basis_features_, *self.output_shape_, self.size)
        return np.tensordot(self.basis.probes(x), grad_coeffs, axes=1)
