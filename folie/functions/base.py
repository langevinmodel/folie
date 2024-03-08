from .._numpy import np
import abc


from sklearn.base import TransformerMixin
from sklearn import linear_model

import scipy.optimize
import sparse

from ..base import _BaseMethodsMixin
from .numdifference import approx_fprime
from .functions_composition import FunctionSum, FunctionTensored


class Function(_BaseMethodsMixin, TransformerMixin):
    r"""
    Base class of all functions that hold spatial dependance of the parameters of the model
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, domain, output_shape=(), **kwargs):
        if output_shape is None:
            output_shape = ()
        self.output_shape_ = np.asarray(output_shape, dtype=int)
        self.output_size_ = np.prod(self.output_shape_)
        self.domain = domain

    def resize(self, new_shape):
        """
        Change the output shape of the function.

        Parameters
        ----------
            new_shape : tuple, array-like
                The new output shape of the function
        """
        self.output_shape_ = np.asarray(new_shape, dtype=int)
        self.output_size_ = np.prod(self.output_shape_)
        return self

    @classmethod
    def __subclasshook__(cls, C):  # Define required minimal interface for duck-typing
        required = ["__call__", "grad_x", "fit", "resize"]
        rtn = True
        for r in required:
            if not any(r in B.__dict__ for B in C.__mro__):
                rtn = NotImplemented
        return rtn

    @property
    def size(self):
        return len(self.coefficients)

    def transform(self, x, *args, **kwargs):
        r"""Transforms the input data."""
        pass

    def fit(self, x, y=None, **kwargs):
        return self

    def transform_x(self, x, **kwargs):
        r"""Gradient of the function with respect to input data.
        Implemented by finite difference.
        """
        return approx_fprime(x, self.__call__, self.output_shape_)

    def transform_xx(self, x, *args, **kwargs):  # TODO: Check
        r"""Hessian of the function with respect to input data.
        Implemented by finite difference.
        """
        return approx_fprime(x, self.transform_x, self.output_shape_ + (self.domain.dim,))

    def __call__(self, x, *args, **kwargs):
        return self.transform(x[:, : self.domain.dim], *args, **kwargs).reshape((-1, *self.output_shape_))

    def grad_x(self, x, *args, **kwargs):
        return self.transform_x(x[:, : self.domain.dim], *args, **kwargs).reshape((-1, *self.output_shape_, len(x[0, : self.domain.dim])))

    def hessian_x(self, x, *args, **kwargs):
        return self.transform_xx(x[:, : self.domain.dim], *args, **kwargs).reshape((-1, *self.output_shape_, len(x[0, : self.domain.dim]), len(x[0, : self.domain.dim])))

    def __add__(self, other):
        return FunctionSum([self, other])

    def __mul__(self, other):
        return FunctionTensored([self, other])

    def __rmul__(self, other):
        return FunctionTensored([self, other])

    def differentiate(self):
        """
        If available differentiate the function with respect to x and return the resulting function
        """
        raise NotImplementedError

    def copy(self):
        r"""Makes a deep copy of this function.

        Returns
        -------
        copy
            A new copy of this model.
        """
        import copy

        return copy.deepcopy(self)


class ParametricFunction(Function):
    r"""Base class of all parametric functions. A function allows for the expression of f(x,coefficients)
    Coefficients are hold by the class and should be given a priori.
    This is mainly a overlay to get a common interface to different python librairies.
    """

    def __init__(self, domain, output_shape=(), coefficients=None, **kwargs):
        super().__init__(domain, output_shape)
        if coefficients is None:
            self._coefficients = np.random.randn(self.n_functions_features_, self.output_size_)  # Initialize to random value
        else:
            self.coefficients = coefficients

    @classmethod
    def __subclasshook__(cls, C):  # Define required minimal interface for duck-typing
        required = ["__call__", "grad_x", "grad_coeffs", "fit", "coefficients", "resize"]
        rtn = True
        for r in required:
            if not any(r in B.__dict__ for B in C.__mro__):
                rtn = NotImplemented
        return rtn

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

        Fx = self.grad_coeffs(x, **kwargs).reshape((x.shape[0] * self.output_size_, -1))
        if isinstance(Fx, sparse.SparseArray):
            Fx = Fx.tocsr()
        reg = estimator.fit(Fx, y, sample_weight=sample_weight)
        self.coefficients = reg.coef_
        self.fitted_ = True
        return self

    def transform_coeffs(self, x, *args, **kwargs):
        init_coeffs = self.coefficients.copy()

        def f_coeffs(c, *args, **kwargs):
            self.coefficients = c
            return self.transform(*args, **kwargs)

        fprime = scipy.optimize.approx_fprime(self.coefficients, f_coeffs, x, *args, **kwargs)
        self.coefficients = init_coeffs
        return fprime
        # raise NotImplementedError  # TODO: Check implementation

    def grad_coeffs(self, x, *args, **kwargs):
        r"""Gradient of the function with respect to the coefficients.

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The gradient
        """
        return self.transform_coeffs(x[:, : self.domain.dim], *args, **kwargs).reshape((x.shape[0], *self.output_shape_, -1))

    def resize(self, new_shape):
        super().resize(new_shape)
        self._coefficients = np.resize(self._coefficients, (self.n_functions_features_, self.output_size_))
        return self

    @property
    def size(self):
        return self.n_functions_features_ * self.output_size_  # Es-ce qu'on ne devrait pas prendre plutot le gcd des deux ou équivalents

    @property
    def shape(self):
        return self.output_shape_

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self._coefficients.ravel()

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._coefficients = vals.reshape((self.n_functions_features_, -1))
