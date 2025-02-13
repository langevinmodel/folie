from .._numpy import np, jacobian
import abc


from sklearn.base import TransformerMixin
from sklearn import linear_model

import scipy.optimize
import sparse

from ..base import _BaseMethodsMixin
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

    def wrap_call(self,x,coeffs, *args, **kwargs):
        self.coefficients=coeffs
        return self(x, *args, **kwargs)

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
            y = np.zeros((x.shape[0], self.output_size_))
        # else:
        #     y = y.ravel()


        def func_wrapped(coeffs):
            self.coefficients=coeffs
            return np.sqrt(((self(x) - y)**2).sum(1))

        # En fait curve_fit c'est assez merdique puisque on appelle pas les paramètres via un array, il faudrait utiliser minimize et définir soit même la loss
        # print(x.shape,self.coefficients)
        res=scipy.optimize.least_squares(func_wrapped, self.coefficients,jac= jacobian(func_wrapped))  # TODO: Plutôt passer par curve_fit ou un truc equivalent
        self.coefficients = res.x
        # print(self.coefficients)
        # Fx=jacobian(self.wrap_call,1)(x, self.coefficients, *args, **kwargs).reshape((x.shape[0] * self.output_size_, -1))

        # if isinstance(Fx, sparse.SparseArray):
        #     Fx = Fx.tocsr()
        # reg = estimator.fit(Fx, y, sample_weight=sample_weight)
        # self.coefficients = reg.coef_
        self.fitted_ = True
        return self

    def __call__(self, x, *args, **kwargs):
        return self.transform(x[:, : self.domain.dim], *args, **kwargs).reshape((-1, *self.output_shape_))

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
