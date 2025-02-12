from .._numpy import np
import abc


from sklearn.base import TransformerMixin
from sklearn import linear_model

import scipy.optimize
import sparse

from ..base import _BaseMethodsMixin
from ._numdifference import approx_fprime
from ..domains import Domain


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

    def fit(self, x, *args, y=None, estimator=linear_model.LinearRegression(copy_X=False, fit_intercept=False), sample_weight=None, **kwargs):
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
        Fx = self.grad_coeffs(x, *args, **kwargs).reshape((x.shape[0] * self.output_size_, -1))
        if isinstance(Fx, sparse.SparseArray):
            Fx = Fx.tocsr()
        reg = estimator.fit(Fx, y, sample_weight=sample_weight)
        self.coefficients = reg.coef_
        self.fitted_ = True
        return self

    def transform_dx(self, x, **kwargs):
        r"""Gradient of the function with respect to input data.
        Implemented by finite difference.
        """
        return approx_fprime(x, self.__call__, self.output_shape_)

    def transform_d2x(self, x, *args, **kwargs):  # TODO: Check
        r"""Hessian of the function with respect to input data.
        Implemented by finite difference.
        """
        return approx_fprime(x, self.transform_dx, self.output_shape_ + (self.domain.dim,))

    def transform_dcoeffs(self, x, *args, **kwargs):
        init_coeffs = self.coefficients.copy()

        def f_coeffs(c, *args, **kwargs):
            self.coefficients = c
            return self.transform(*args, **kwargs)

        fprime = scipy.optimize.approx_fprime(self.coefficients, f_coeffs, x, *args, **kwargs)
        self.coefficients = init_coeffs
        return fprime
        # raise NotImplementedError  # TODO: Check implementation

    def __call__(self, x, *args, **kwargs):
        return self.transform(x[:, : self.domain.dim], *args, **kwargs).reshape((-1, *self.output_shape_))

    def grad_x(self, x, *args, **kwargs):
        return self.transform_dx(x[:, : self.domain.dim], *args, **kwargs).reshape((-1, *self.output_shape_, len(x[0, : self.domain.dim])))

    def hessian_x(self, x, *args, **kwargs):
        return self.transform_d2x(x[:, : self.domain.dim], *args, **kwargs).reshape((-1, *self.output_shape_, len(x[0, : self.domain.dim]), len(x[0, : self.domain.dim])))

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
        return self.transform_dcoeffs(x[:, : self.domain.dim], *args, **kwargs).reshape((x.shape[0], *self.output_shape_, -1))

    def __add__(self, other):
        return FunctionSum([self, other])

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


class ModelOverlay(Function):
    """
    A class that allow to overlay a model and make it be used as a function.

    For example, if a model contain

    .. code-block:: python

        model.drift =  ModelOverlay(model, "_force", output_shape=output_shape_force)

    then we have the following mapping:

        - model.drift(x) -> model._force(x)
        - model.drift.grad_x(x) -> model._force_dx(x)
        - model.drift.hessian_x(x) -> model._force_d2x(x)
        - model.drift.grad_coeffs(x) -> model._force_dcoeffs(x)
        - model.drift.coefficients -> model._force_coefficients

    If any of the [function_name]_d* function is not implemented, it would be replaced by a numerical derivative

    Parameters
    ----------
        model: a python class
            This is the class to to overlay

        function_name: str
            The common part of the function name. The model should contain at least the function [function_name]

        output_shape: tuple or array
            The output shape of the term

    """

    def __init__(self, model, function_name, output_shape=None, **kwargs):
        self.model = model
        domain = Domain.Rd(model.dim)
        # Do some check
        if not hasattr(self.model, function_name):
            raise ValueError("Model does not implement " + function_name + ".")
        self.function_name = function_name
        # Define output shape from model dimension
        if output_shape is None and model.dim <= 1:
            output_shape = ()
        elif output_shape is None:
            raise ValueError("output_shape should be defined.")

        super().__init__(domain, output_shape, **kwargs)

    def transform(self, x, *args, **kwargs):
        return getattr(self.model, self.function_name)(x, *args, **kwargs)

    def transform_dx(self, x, *args, **kwargs):
        if hasattr(self.model, self.function_name + "_dx"):
            return getattr(self.model, self.function_name + "_dx")(x, *args, **kwargs)
        else:
            return super().transform_dx(x, *args, **kwargs)  # If not implemented use finite difference

    def transform_d2x(self, x, *args, **kwargs):
        if hasattr(self.model, self.function_name + "_d2x"):
            return getattr(self.model, self.function_name + "_d2x")(x, *args, **kwargs)
        else:
            return super().transform_d2x(x, *args, **kwargs)  # If not implemented use finite difference

    def transform_dcoeffs(self, x, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        if hasattr(self.model, self.function_name + "_dcoeffs"):
            return getattr(self.model, self.function_name + "_dcoeffs")(x, *args, **kwargs)
        else:
            return super().transform_dcoeffs(x, *args, **kwargs)  # If not implemented use finite difference

    @property
    def coefficients(self):
        """Access the coefficients"""
        return getattr(self.model, "coefficients" + self.function_name)

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        setattr(self.model, "coefficients" + self.function_name, vals)

    @property
    def size(self):
        return np.prod(self.coefficients.shape)  # Es-ce qu'on ne devrait pas prendre plutot le gcd des deux ou équivalents


class FunctionSum:
    """
    Return the sum of function
    """

    def __init__(self, functions):
        self.functions_set = functions
        for fu in self.functions_set:
            if fu.output_shape_ != self.functions_set[0].output_shape_:
                raise ValueError("Cannot sum function with different output shape")
        # Do some samity check on the output of each functions

    def resize(self, new_shape):
        super().resize(new_shape)
        for fu in self.functions_set:
            fu.resize(new_shape)
        return self

    def __add__(self, other):
        if type(other) is FunctionSum:
            self.functions_set.extend(other.functions_set)
            self.factors_set.extend(other.factors_set)
        else:
            self.functions_set.append(other)
            self.factors_set.append(1.0)
        return self

    def __call__(self, X, *args, **kwargs):
        return np.add.reduce([fu(X) for fu in self.functions_set])

    def grad_x(self, X, **kwargs):
        return np.add.reduce([fu.grad_x(X) for fu in self.functions_set])

    def grad_coeffs(self, X, **kwargs):
        return np.concatenate([fu.grad_coeffs(X) for fu in self.functions_set], axis=-1)

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate([fu.coefficients for fu in self.functions_set])

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        curr_size = 0
        for fu in self.functions_set:
            fu.coefficients = vals.ravel[curr_size : curr_size + fu.size]
            curr_size += fu.size

    @property
    def size(self):
        return np.sum([fu.size for fu in self.functions_set])

    @property
    def shape(self):
        return self.functions_set[0].output_shape_

    @property
    def n_output_features_(self):
        return np.sum([fu.n_output_features_ for fu in self.functions_set])
