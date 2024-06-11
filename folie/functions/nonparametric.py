from .base import Function
from .._numpy import np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted, check_array
from .base import ParametricFunction


class KernelFunction(Function):
    def __init__(self, gamma, domain, kernel="rbf", output_shape=()):
        super().__init__(domain, output_shape)
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, x, y=None, gamma_range=None, **kwargs):
        """Set reference frame associated with function values

        Parameters
        ----------
        X : array-like of shape (n_references, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : array-like of shape (n_references, )
            The target values of the function to interpolate.
        gamma_range: iterable
            List of gamma values to try for optimization

        Returns
        -------
        self : object
            Returns self.
        """
        if y is None:
            y = np.zeros((x.shape[0], self.output_size_))
        else:
            y = y.reshape((x.shape[0], -1))
        self.ref_X = check_array(x, accept_sparse=True)
        self.ref_f = y
        self.fitted_ = True
        if hasattr(gamma_range, "__iter__"):
            self.gamma = self._optimize_gamma(gamma_range)
        return self  # `fit` should always return `self`

    def transform(self, X, *args, **kwargs):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        s : ndarray, shape (n_samples,)
            Value of path variable along the path.
        z : ndarray, shape (n_samples,)
            Value of the distance to the path.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "fitted_")
        K = pairwise_kernels(self.ref_X, X, metric=self.kernel, gamma=self.gamma)
        s = (K[..., None] * self.ref_f[:, None, ...]).sum(axis=0) / K.sum(axis=0)[:, None]
        return s

    def _optimize_gamma(self, gamma_values):
        # Select specific value of gamma from the range of given gamma_values
        # by minimizing mean-squared error in leave-one-out cross validation
        mse = np.empty_like(gamma_values, dtype=float)
        for i, gamma in enumerate(gamma_values):
            K = pairwise_kernels(self.ref_X, self.ref_X, metric=self.kernel, gamma=gamma)
            np.fill_diagonal(K, 0)  # leave-one-out
            Ky = K * self.ref_f[..., np.newaxis]
            y_pred = Ky.sum(axis=0) / K.sum(axis=0)
            mse[i] = ((y_pred - self.ref_f) ** 2).mean()

        return gamma_values[np.nanargmin(mse)]

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.array([self.gamma])

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.gamma = vals.ravel()[0]


class sklearnWrapper(Function):
    """
    Wraps sklearn predictor as functions. Allow to use non parametric estimator for fit.
    """

    def __init__(self, estimator, domain, output_shape=(), expose_params=[]):
        """
        expose_params is a list of key for the parameters of the estimator to be exposed for optimisation
        """
        super().__init__(domain, output_shape)
        self.estimator = estimator
        self.exposed_params = expose_params
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The estimator does not have predict method")

    def fit(self, x, y=None, **kwargs):
        if y is None:
            y = np.zeros((x.shape[0], self.output_size_))
        else:
            y = y.reshape((x.shape[0], -1))
        self.estimator = self.estimator.fit(x, y)
        self.fitted_ = True
        return self

    def transform(self, X, *args, **kwargs):
        return self.estimator.predict(X)

    @property
    def coefficients(self):
        """Access the coefficients"""
        est_params = self.estimator.get_params()
        coeffs = np.array([])
        for key in self.exposed_params:
            if key in est_params:
                np.concatenate((coeffs, est_params[key].ravel()), axis=0)
            else:
                np.concatenate((coeffs, getattr(self.estimator, key).ravel()), axis=0)
        return coeffs

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        curr_ind = 0
        est_params = self.estimator.get_params()
        for key in self.exposed_params:
            if key in est_params:
                shape_params = est_params[key].shape
                size_param = int(np.prod(shape_params))
                self.estimator.set_params({key: (vals.ravel()[curr_ind : curr_ind + size_param]).reshape(shape_params)})
            else:
                shape_params = getattr(self.estimator, key).shape
                size_param = int(np.prod(shape_params))
                setattr(self.estimator, key, (vals.ravel()[curr_ind : curr_ind + size_param]).reshape(shape_params))
            curr_ind += size_param


class sklearnTransformer(ParametricFunction):
    """
    Take any sklearn transformer and build a fonction from it
    f(x,c) = c*transform(x)

    transformer can also be a DiffusionMap object from pyDiffMap, but it should be fitted a priori
    """

    def __init__(self, transformer, domain, output_shape=(), coefficients=None):
        super().__init__(domain, output_shape, coefficients)
        self.transformer = transformer
        if not check_is_fitted(self.transformer):
            self.transformer = self.transformer.fit(domain.cube)
        self.n_functions_features_ = self.transformer.transform(domain.cube).shape[1]

    def transform(self, x, *args, **kwargs):
        return self.transformer.transform(x) @ self._coefficients

    def transform_dcoeffs(self, x, **kwargs):
        transform_dcoeffs = np.eye(self.size).reshape(self.n_functions_features_, *self.output_shape_, self.size)
        return np.tensordot(self.transformer.transform(x), transform_dcoeffs, axes=1)
