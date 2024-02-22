from .base import Function
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted, check_array


class sklearnWrapper(Function):
    """
    Wraps sklearn predictor as functions. Allow to use non parametric estimator for fit.
    """

    def __init__(self, estimator, output_shape=()):
        super().__init__(output_shape)
        # TODO: add possibility to expose some of the hyperparameters for MLE optimisation
        self.estimator = estimator

    def fit(self, x, y=None, **kwargs):
        if y is None:
            y = np.zeros((x.shape[0], self.output_size_))
        else:
            y = y.reshape((x.shape[0], -1))
        self.estimator = self.estimator.fit(x, y)
        return self

    def transform(self, X, *args, **kwargs):
        return self.estimator.predict(X)


class KernelFunction(Function):
    def __init__(self, gamma, kernel="rbf", output_shape=()):
        super().__init__(output_shape)
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, x, y=None, **kwargs):
        """Set reference frame associated with function values

        Parameters
        ----------
        X : array-like of shape (n_references, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : array-like of shape (n_references, )
            The target values of the function to interpolate.

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
        self.is_fitted_ = True
        if hasattr(self.gamma, "__iter__"):
            self.gamma = self._optimize_gamma(self.gamma)
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
        check_is_fitted(self, "is_fitted_")
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
