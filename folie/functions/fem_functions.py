from .base import ParametricFunction
from .._numpy import np
from ..data import stats_from_input_data
import sparse
from sklearn import linear_model


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
        cells = self.basis.mesh.element_finder(mapping=self.basis.mapping)(*(X.T))  # Change the element finder
        # Find a way to exclude out of mesh elements, we can define an outside elements that is a constant
        loc_x = self.basis.mapping.invF(X.T[:, :, np.newaxis], tind=cells)
        return cells, loc_x[..., 0].T

    def fit(self, x, y=None, estimator=linear_model.LinearRegression(copy_X=False, fit_intercept=False), sample_weight=None, **kwargs):
        """
        Fit coefficients of the function using linear regression.
        Use as features the derivative of the function with respect to the coefficients

        Parameters
        ----------

            X : {array-like} of shape (n_samples, dim)
            Point of evaluation of the training data

            y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.


            estimator: sklearn compatible estimator
            Defaut to sklearn.linear_model.LinearRegression(copy_X=False, fit_intercept=False) but any compatible estimator can be used.
            Estimator should have a coef_ attibutes after fitting

        """
        if y is None:
            y = np.zeros((x.shape[0] * self.output_size_))
        else:
            y = y.ravel()

        Fx = self.grad_coeffs(x, **kwargs)
        reg = estimator.fit(Fx.reshape((x.shape[0] * self.output_size_, -1)).tocsr(), y, sample_weight=sample_weight)
        self.coefficients = reg.coef_
        self.fitted_ = True
        return self

    def transform(self, x, *args, **kwargs):
        try:
            cells = kwargs["cells_idx"]
            loc_x = kwargs["loc_x"]
        except KeyError:
            cells, loc_x = self.preprocess_traj(x)
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
            cells, loc_x = self.preprocess_traj(x)
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
            cells, loc_x = self.preprocess_traj(x)
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
