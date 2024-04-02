from .._numpy import np


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

    def differentiate(self):
        return FunctionSum([fu.differentiate() for fu in self.functions_set])

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


class FunctionTensored:
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
        self.fitted_ = True
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


class FunctionOffset:
    """
    A composition Function to represent f(x)+g(x)v where
    """

    def __init__(self, f, g):
        self._f = f
        self._g = g
        # super().__init__(output_shape=self.f.output_shape_)

    def fit(self, x, v, y=None, *args, **kwargs):
        if y is not None:
            gv = np.einsum("t...h,th-> t...", self.g(x, *args, **kwargs).reshape((*y.shape, v.shape[1])), v)
            y -= gv
        self._f = self._f.fit(x, y)
        # Mais du coup g n'est pas fit
        return self

    def __call__(self, x, v, *args, **kwargs):
        fx = self._f(x, *args, **kwargs)
        return fx + np.einsum("t...h,th-> t...", self._g(x, *args, **kwargs).reshape((*fx.shape, v.shape[1])), v)

    def grad_x(self, x, v, *args, **kwargs):
        dfx = self._f.grad_x(x, *args, **kwargs)
        return dfx + np.einsum("t...he,th-> t...e", self._g.grad_x(x, *args, **kwargs).reshape((*dfx.shape[:-1], v.shape[1], dfx.shape[-1])), v)

    def hessian_x(self, x, v, *args, **kwargs):
        ddfx = self._f.hessian_x(x, *args, **kwargs)
        return ddfx + np.einsum("t...hef,th-> t...ef", self._g.hessian_x(x, *args, **kwargs).reshape((*ddfx.shape[:-2], v.shape[1], *ddfx.shape[-2:])), v)

    def __getattr__(self, item):  # Anything else should be passed to f
        if item == "f":
            return self._f
        elif item == "g":
            return self._g
        else:
            return getattr(self._f, item)

    def __setattr__(self, item, value):
        if item.startswith("_"):
            # If it's a private attribute, handle it normally
            super().__setattr__(item, value)
        else:
            # If it's not a private attribute, delegate the assignment to the inner object
            setattr(self._f, item, value)
