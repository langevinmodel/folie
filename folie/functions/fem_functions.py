from .base import ParametricFunction
from .._numpy import np
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
        # phis = np.array([self.basis.elem.gbasis(self.basis.mapping, loc_x.T[..., None], k, tind=cells)[0] for k in range(self.basis.Nbfun)])
        # probes_matrix = sparse.COO(
        #     np.array(
        #         [
        #             np.tile(np.arange(x.shape[0]), self.basis.Nbfun),
        #             self.basis.element_dofs[:, cells].flatten(),
        #         ]
        #     ),
        #     phis.flatten(),
        #     shape=(x.shape[0], self.basis.N),
        # )
        # return probes_matrix.tocsr() @ self._coefficients

        res = 0.0
        for k in range(self.basis.Nbfun):
            res += self.basis.elem.gbasis(self.basis.mapping, loc_x.T[..., None], k, tind=cells)[0] * self._coefficients[self.basis.element_dofs[k, cells], ...]
        return res

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
            transform_coeffs = sparse.COO.from_numpy(np.eye(self.size).reshape((self.n_functions_features_, self.output_size_, self.size)))
            return sparse.einsum("nb,bsc->nsc", probes_matrix, transform_coeffs)


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

    def transform_x(self, x, *args, **kwargs):
        _, dim = x.shape
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.domain.localize_data(x)
        return linear_interpolation_x(cells, loc_x, self.coefficients)

    def transform_coeffs(self, x, *args, **kwargs):
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.domain.localize_data(x)
        return linear_element_gradient(cells, loc_x, self.size)
