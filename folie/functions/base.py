import numpy as np
import abc


from sklearn.base import TransformerMixin

from ..base import _BaseMethodsMixin


class Function(_BaseMethodsMixin, TransformerMixin):
    r"""Base class of all functions. A function allows for the expression of f(x,coefficients)
    Coefficients are hold by the class and should be given a priori.
    This is mainly a overlay to get a common interface to different python librairies.
    """

    def __init__(self, output_shape=(), **kwargs):
        if output_shape is None:
            output_shape = ()
        self.output_shape_ = np.asarray(output_shape, dtype=int)
        self.output_size_ = np.prod(self.output_shape_)
        self._coefficients = None

    def resize(self, new_shape):
        """
        Change the output shape of the function.

        Parameters
        ----------
            new_shape : tuple, array-like
                The new output shape of the function
            force_reshape : bool, defaut=False
                Whatever to force reshape if current shape of coefficients is incompatible with new shape
                If True new zeros coefficients will be added as needed
        """
        self.output_shape_ = new_shape
        self.output_size_ = np.prod(self.output_shape_)
        return self

    def fit(self, x, y=None):
        return self

    @abc.abstractmethod
    def transform(self, x, **kwargs):
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

    def grad_x(self, x, **kwargs):
        r"""Gradient of the function with respect to input data.
        Implemented by finite difference.

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The gradient
        """
        # TODO: A faire avec scipy approx_fprime et prendre en compte le ND
        return (self(x + self.h) - self(x - self.h)) / (2 * self.h)

    @abc.abstractmethod
    def grad_coeffs(self, x, **kwargs):
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

    def __call__(self, x, *args, **kwargs):
        return self.transform(x, *args, **kwargs).reshape(-1, *self.output_shape_)

    def __add__(self, other):
        return FunctionSum([self, other])

    def __mul__(self, other):
        return FunctionTensored([self, other])

    def __rmul__(self, other):
        return FunctionTensored([self, other])

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self._coefficients.ravel()

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._coefficients = vals

    @property
    def is_linear(self) -> bool:
        """Return True is the model is linear in its parameters"""
        return False

    @property
    def size(self):
        return np.size(self._coefficients)

    @property
    def shape(self):
        return self.output_shape_

    def copy(self):
        r"""Makes a deep copy of this function.

        Returns
        -------
        copy
            A new copy of this model.
        """
        import copy

        return copy.deepcopy(self)


class FunctionFromBasis(Function):
    def __init__(self, output_shape=(), basis=None):
        super().__init__(output_shape)
        self.basis = basis

    def fit(self, x, y=None):
        if self.basis is not None:
            self.basis.fit(x)
            self.n_basis_features_ = self.basis.n_output_features_
        self.coefficients = np.zeros((self.n_basis_features_, self.output_size_))
        return self

    def resize(self, new_shape):
        super().resize(new_shape)
        self._coefficients = np.resize(self._coefficients, (self.n_basis_features_, self.output_size_))
        return self

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self._coefficients.ravel()

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._coefficients = vals.reshape((self.n_basis_features_, self.output_size_))

    def transform(self, x, **kwargs):
        return self.basis(x) @ self._coefficients

    def grad_x(self, x, **kwargs):
        _, dim = x.shape
        return np.einsum("nbd,bs->nsd", self.basis.deriv(x), self._coefficients).reshape(-1, *self.output_shape_, dim)

    def grad_coeffs(self, x, **kwargs):
        grad_coeffs = np.eye(self.size).reshape(self.n_basis_features_, *self.output_shape_, self.size)
        return np.tensordot(x, grad_coeffs, axes=1)

    @property
    def is_linear(self) -> bool:
        """Return True is the model is linear in its parameters"""
        return True


class FunctionSum(Function):
    """
    Return the sum of function
    """

    def __init__(self, functions):
        self.functions_set = functions
        self.output_shape_ = self.functions_set[0].output_shape_
        for fu in self.functions_set:
            if fu.output_shape_ != self.output_shape_:
                raise ValueError("Cannot sum function with different output shape")
        self.output_size_ = np.prod(self.output_shape_)
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

    def fit(self, data):
        for fu in self.functions_set:
            fu.fit(data)
        self.n_output_features_ = np.sum([fu.n_output_features_ for fu in self.functions_set])
        self.dim_out_basis = 1
        return self

    def transform(self, X, **kwargs):
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
    def is_linear(self) -> bool:
        """Return True is the model is linear in its parameters"""
        return False

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

    def fit(self, x, y=None):
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
        self.g.fit(x, y)
        self.f.fit(self.g(x), y)

        self.n_output_features_ = self.f.n_output_features_
        self.n_input_dim_ = self.g.n_input_dim_

        return self

    def transform(self, x, **kwargs):
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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
            By default, it is assumed to be tensord product over one dimensionnal function

        """
        self.functions_set = functions
        if input_dims is None:
            input_dims = (1,) * len(functions)
        if len(input_dims) != len(self.functions_set):
            raise ValueError("Incoherent decompositions of input dimensions")
        bounds = np.insert(np.cumsum(input_dims), 0, 0)
        self.input_dims_slice = [slice(int(bounds[n]), int(bounds[n + 1])) for n in range(len(self.functions_set))]

    def fit(self, x, y=None):
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
