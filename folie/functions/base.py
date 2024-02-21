import numpy as np
import abc


from sklearn.base import TransformerMixin
from sklearn import linear_model

from ..base import _BaseMethodsMixin
from .numdifference import approx_fprime


class Function(_BaseMethodsMixin, TransformerMixin):
    r"""
    Base class of all functions that hold spatial dependance of the parameters of the model
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, output_shape=(), dim_x=None, **kwargs):
        if output_shape is None:
            output_shape = ()
        self.output_shape_ = np.asarray(output_shape, dtype=int)
        self.output_size_ = np.prod(self.output_shape_)
        self.dim_x = dim_x

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

    @abc.abstractmethod
    def transform(self, x, *args, **kwargs):
        r"""Transforms the input data."""

    def fit(self, x, y=None, **kwargs):
        return self

    def transform_x(self, x, **kwargs):
        r"""Gradient of the function with respect to input data.
        Implemented by finite difference.
        """
        return approx_fprime(x, self.__call__, self.output_shape_)

    def __call__(self, x, *args, **kwargs):
        return self.transform(x[:, : self.dim_x], *args, **kwargs).reshape(-1, *self.output_shape_)

    def grad_x(self, x, *args, **kwargs):
        return self.transform_x(x[:, : self.dim_x], *args, **kwargs).reshape(-1, *self.output_shape_, len(x[0, : self.dim_x]))

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


class FunctionSum(Function):
    """
    Return the sum of function
    """

    def __init__(self, functions):
        self.functions_set = functions
        super().__init__(output_shape=self.functions_set[0].output_shape_)
        for fu in self.functions_set:
            if fu.output_shape_ != self.output_shape_:
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

    def differentiate(self):
        return FunctionSum([fu.differentiate() for fu in self.functions_set])

    def fit(self, data, **kwargs):
        for fu in self.functions_set:
            fu.fit(data, **kwargs)
        self.n_output_features_ = np.sum([fu.n_output_features_ for fu in self.functions_set])
        self.dim_out_basis = 1
        return self

    def transform(self, X, *args, **kwargs):
        return np.add.reduce([fu.transform(X) for fu in self.functions_set])

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


class FunctionComposition(Function):
    r"""
    Composition operation to evaluate :math:`(f \circ g)(x) = f(g(x))`, where
    :math:`f` and :math:`g` are functions
    """

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def fit(self, x, y=None, **kwargs):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        self.g.fit(x, y, **kwargs)  # TODO: Ne marche pas, il faudrait faire f^-1(y)
        self.f.fit(self.g(x), y, **kwargs)

        self.n_output_features_ = self.f.n_output_features_

        return self

    def transform(self, x, *args, **kwargs):
        r"""Transforms the input data.

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The transformed data
        """
        return

    def grad_x(self, x, **kwargs):
        r"""Gradient of the function with respect to input data

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The transformed data
        """

    def hessian_x(self, x, **kwargs):
        """
        Hessien of the function with respect to input data
        """

    def grad_coeffs(self, x, **kwargs):
        r"""Transforms the input data.

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The transformed data
        """


class FunctionTensored(Function):
    r"""
    Tensor operation to evaluate :math:`(f_1 \otimes f_2 \otimes f_3)(x,y,z) = f_1(x)f_2(y)f_3(z)`, where
    :math:`f_1`, :math:`f_2` and :math:`f_3` are functions
    """

    def __init__(self, functions, input_dims=None):
        """
        Parameters
        ----------
        functions : list
            The list of functions in the tensor product

        input_dims : tuple
            The decomposition of the dimension of X into (x,y,z,..).
            By default, it is assumed to be tensor product over one dimensionnal function

        """
        self.functions_set = functions
        if input_dims is None:
            input_dims = (1,) * len(functions)
        if len(input_dims) != len(self.functions_set):
            raise ValueError("Incoherent decompositions of input dimensions")
        bounds = np.insert(np.cumsum(input_dims), 0, 0)
        self.input_dims_slice = [slice(int(bounds[n]), int(bounds[n + 1])) for n in range(len(self.functions_set))]

    def fit(self, x, y=None, **kwargs):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """

        self.n_features_in_ = x.shape[1]

        # First fit all libs provided below
        fitted_fun = [fu.fit(x[..., self.input_dims_slice[i]], y) for i, fu in enumerate(self.functions_set)]

        # Calculate the sum of output features
        output_sizes = [lib.n_output_features_ for lib in fitted_fun]
        self.n_output_features_ = 1
        for osize in output_sizes:
            self.n_output_features_ *= osize

        # Save fitted libs
        self.functions_set = fitted_fun

        return self

    def transform(self, x, *args, **kwargs):
        r"""Transforms the input data.

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The transformed data
        """
        return

    def grad_x(self, x, **kwargs):
        r"""Gradient of the function with respect to input data

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The transformed data
        """
        raise NotImplementedError

    def hessian_x(self, x, **kwargs):
        """
        Hessien of the function with respect to input data
        """
        raise NotImplementedError

    def grad_coeffs(self, x, **kwargs):
        r"""Transforms the input data."""
        raise NotImplementedError


class FunctionOffset(Function):
    """
    A composition Function to represent f(x)+g(x)v where
    """

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def fit(self, x, v, y=None, **kwargs):
        gv = np.einsum("t...h,th-> t...", self.g(x), v)
        if y is None:
            y = gv
        self.f = self.f.fit(x, y - gv)
        # TODO: Check if g have correct shape
        return self

    def transform(self, x, v, *args, **kwargs):
        return self.f(x, *args, **kwargs) + np.einsum("t...h,th-> t...", self.g(x), v)

    def grad_x(self, x, v):
        return self.f.grad_x(x) + np.einsum("t...he,th-> t...e", self.g.grad_x(x), v)

    def __getattr__(self, item):  # Anything else should be passed to f
        return getattr(self.f, item)


class ParametricFunction(Function):
    r"""Base class of all parametric functions. A function allows for the expression of f(x,coefficients)
    Coefficients are hold by the class and should be given a priori.
    This is mainly a overlay to get a common interface to different python librairies.
    """

    def __init__(self, output_shape=(), coefficients=None, **kwargs):
        super().__init__(output_shape)
        if coefficients is None:
            self._coefficients = np.zeros(output_shape)
            self.n_functions_features_ = 1
        else:
            self.n_functions_features_ = coefficients.shape[0]
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

        Fx = self.grad_coeffs(x)
        reg = estimator.fit(Fx.reshape((x.shape[0] * self.output_size_, -1)), y, sample_weight=sample_weight)
        self.coefficients = reg.coef_
        return self

    @abc.abstractmethod
    def transform_coeffs(self, x, *args, **kwargs):
        pass  # TODO: Implement finite difference with scipy?

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
        return self.transform_coeffs(x[:, : self.dim_x], *args, **kwargs).reshape(-1, *self.output_shape_, self.size)

    def resize(self, new_shape):
        super().resize(new_shape)
        self._coefficients = np.resize(self._coefficients, (self.n_functions_features_, self.output_size_))
        return self

    @property
    def size(self):
        return self.n_functions_features_ * self.output_size_

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
        self._coefficients = vals.reshape((self.n_functions_features_, self.output_size_))
