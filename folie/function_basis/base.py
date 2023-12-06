"""
The code in this file is adapted from deeptime (https://github.com/deeptime-ml/deeptime/blob/main/deeptime/base.py)
"""

import numpy as np
import abc

from functools import wraps
from itertools import repeat

from typing import Optional
from typing import Sequence

import scipy.interpolate
import scipy.stats


from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import _BaseMethodsMixin


def x_sequence_or_item(wrapped_func):
    """Allow a feature library's method to handle list or item inputs."""

    @wraps(wrapped_func)
    def func(self, x, *args, **kwargs):
        if isinstance(x, Sequence):
            xs = [AxesArray(xi, comprehend_axes(xi)) for xi in x]
            result = wrapped_func(self, xs, *args, **kwargs)
            if isinstance(result, Sequence):  # e.g. transform() returns x
                return [AxesArray(xp, comprehend_axes(xp)) for xp in result]
            return result  # e.g. fit() returns self
        else:
            if not sparse.issparse(x):
                x = AxesArray(x, comprehend_axes(x))

                def reconstructor(x):
                    return x

            else:  # sparse arrays
                reconstructor = type(x)
                axes = comprehend_axes(x)
                wrap_axes(axes, x)
            result = wrapped_func(self, [x], *args, **kwargs)
            if isinstance(result, Sequence):  # e.g. transform() returns x
                return reconstructor(result[0])
            return result  # e.g. fit() returns self

    return func


class Basis(_BaseMethodsMixin, TransformerMixin):
    r"""Base class of all transformers."""

    # Force subclasses to implement this
    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        r"""Transforms the input data.

        Parameters
        ----------
        data : array_like
            Input data.

        Returns
        -------
        transformed : array_like
            The transformed data
        """

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def __add__(self, other):
        return ConcatBasis([self, other])

    def __mul__(self, other):
        return TensoredBasis([self, other])

    def __rmul__(self, other):
        return TensoredBasis([self, other])

    @property
    def size(self):
        check_is_fitted(self)
        return self.n_output_features_


class ConcatBasis(Basis):
    """Concatenate multiple libraries into one library. All settings
    provided to individual libraries will be applied.

    Parameters
    ----------
    libraries : list of libraries
        Library instances to be applied to the input matrix.

    Attributes
    ----------
    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the sum of the numbers of output features for each of the
        concatenated libraries.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import FourierLibrary, CustomLibrary
    >>> from pysindy.feature_library import ConcatLibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib_custom = CustomLibrary(library_functions=functions)
    >>> lib_fourier = FourierLibrary()
    >>> lib_concat = ConcatLibrary([lib_custom, lib_fourier])
    >>> lib_concat.fit()
    >>> lib.transform(x)
    """

    def __init__(
        self,
        libraries: list,
    ):
        self.libraries = libraries

    @x_sequence_or_item
    def fit(self, x_full, y=None):
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
        n_features = x_full[0].shape[x_full[0].ax_coord]
        self.n_features_in_ = n_features

        # First fit all libs provided below
        fitted_libs = [lib.fit(x_full, y) for lib in self.libraries]

        # Calculate the sum of output features
        self.n_output_features_ = sum([lib.n_output_features_ for lib in fitted_libs])

        # Save fitted libs
        self.libraries = fitted_libs

        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data with libs provided below.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.

        """
        for lib in self.libraries:
            check_is_fitted(lib)

        xp_full = []
        for x in x_full:
            feature_sets = [lib.transform([x])[0] for lib in self.libraries]
            xp = np.concatenate(feature_sets, axis=feature_sets[0].ax_coord)

            xp = AxesArray(xp, comprehend_axes(xp))
            xp_full.append(xp)
        return xp_full

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        feature_names = list()
        for lib in self.libraries:
            lib_feat_names = lib.get_feature_names(input_features)
            feature_names += lib_feat_names
        return feature_names

    def calc_trajectory(self, diff_method, x, t):
        return self.libraries[0].calc_trajectory(diff_method, x, t)


class TensoredBasis(Basis):
    """Tensor multiple libraries together into one library. All settings
    provided to individual libraries will be applied.

    Parameters
    ----------
    libraries : list of libraries
        Library instances to be applied to the input matrix.

    inputs_per_library_ : Sequence of Sequences of ints (default None)
        list that specifies which input indexes should be passed as
        inputs for each of the individual feature libraries.
        length must equal the number of feature libraries.  Default is
        that all inputs are used for every library.

    Attributes
    ----------
    libraries_ : list of libraries
        Library instances to be applied to the input matrix.

    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the numbers of output features for each of the
        libraries that were tensored together.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import FourierLibrary, CustomLibrary
    >>> from pysindy.feature_library import TensoredLibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib_custom = CustomLibrary(library_functions=functions)
    >>> lib_fourier = FourierLibrary()
    >>> lib_tensored = lib_custom * lib_fourier
    >>> lib_tensored.fit(x)
    >>> lib_tensored.transform(x)
    """

    def __init__(
        self,
        libraries: list,
        inputs_per_library: Optional[Sequence[Sequence[int]]] = None,
    ):
        self.libraries = libraries
        self.inputs_per_library = inputs_per_library

    def _combinations(self, lib_i, lib_j):
        """
        Compute combinations of the numerical libraries.

        Returns
        -------
        lib_full : All combinations of the numerical library terms.
        """
        # the shape here should be fixed with ax_coord....
        shape = np.array(lib_i.shape)
        shape[lib_i.ax_coord] = lib_i.shape[lib_i.ax_coord] * lib_j.shape[lib_j.ax_coord]
        lib_full = np.reshape(
            lib_i[..., :, np.newaxis] * lib_j[..., np.newaxis, :],
            shape,
        )

        return lib_full

    def _name_combinations(self, lib_i, lib_j):
        """
        Compute combinations of the library feature names.

        Returns
        -------
        lib_full : All combinations of the library feature names.
        """
        lib_full = []
        for i in range(len(lib_i)):
            for j in range(len(lib_j)):
                lib_full.append(lib_i[i] + " " + lib_j[j])
        return lib_full

    def _set_inputs_per_library(self, inputs_per_library):
        """
        Extra function to make building a GeneralizedLibrary object easier
        """
        self.inputs_per_library = inputs_per_library

    @x_sequence_or_item
    def fit(self, x_full, y=None):
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
        n_features = x_full[0].shape[x_full[0].ax_coord]

        self.n_features_in_ = n_features

        # If parameter is not set, use all the inputs
        if self.inputs_per_library is None:
            self.inputs_per_library = list(repeat(list(range(n_features)), len(self.libraries)))

        # First fit all libs provided below
        fitted_libs = [lib.fit([x[..., _unique(self.inputs_per_library[i])] for x in x_full], y) for i, lib in enumerate(self.libraries)]

        # Calculate the sum of output features
        output_sizes = [lib.n_output_features_ for lib in fitted_libs]
        self.n_output_features_ = 1
        for osize in output_sizes:
            self.n_output_features_ *= osize

        # Save fitted libs
        self.libraries = fitted_libs

        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data with libs provided below.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.

        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            xp = []
            for i in range(len(self.libraries)):
                lib_i = self.libraries[i]
                if self.inputs_per_library is None:
                    xp_i = lib_i.transform([x])[0]
                else:
                    xp_i = lib_i.transform([x[..., _unique(self.inputs_per_library[i])]])[0]

                for j in range(i + 1, len(self.libraries)):
                    lib_j = self.libraries[j]
                    xp_j = lib_j.transform([x[..., _unique(self.inputs_per_library[j])]])[0]

                    xp.append(self._combinations(xp_i, xp_j))

            xp = np.concatenate(xp, axis=xp[0].ax_coord)
            xp = AxesArray(xp, comprehend_axes(xp))
            xp_full.append(xp)
        return xp_full

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        feature_names = list()
        for i in range(len(self.libraries)):
            lib_i = self.libraries[i]
            if input_features is None:
                input_features_i = ["x%d" % k for k in _unique(self.inputs_per_library[i])]
            else:
                input_features_i = np.asarray(input_features)[_unique(self.inputs_per_library[i])].tolist()
            lib_i_feat_names = lib_i.get_feature_names(input_features_i)
            for j in range(i + 1, len(self.libraries)):
                lib_j = self.libraries[j]
                if input_features is None:
                    input_features_j = ["x%d" % k for k in _unique(self.inputs_per_library[j])]
                else:
                    input_features_j = np.asarray(input_features)[_unique(self.inputs_per_library[j])].tolist()
                lib_j_feat_names = lib_j.get_feature_names(input_features_j)
                feature_names += self._name_combinations(lib_i_feat_names, lib_j_feat_names)
        return feature_names

    def calc_trajectory(self, diff_method, x, t):
        return self.libraries[0].calc_trajectory(diff_method, x, t)


def _unique(s: Sequence) -> Sequence:
    """Remove duplicates, preserving order when python > 3.7"""
    return list(dict.fromkeys(s))


class BasisCombiner(Basis):  # TODO A mixer avec ConcatBasis
    """
    Allow to combine features to build composite basis
    """

    def __init__(self, *basis):
        self.basis_set = basis
        self.const_removed = np.any([b.const_removed for b in self.basis_set])  # Check if one of the basis set have the constant removed

    def fit(self, describe_result):
        if isinstance(describe_result, np.ndarray):
            describe_result = scipy.stats.describe(describe_result)
        for b in self.basis_set:
            b.fit(describe_result)
        self.n_output_features_ = np.sum([b.n_output_features_ for b in self.basis_set])
        self.dim_out_basis = 1
        return self

    def transform(self, X, **kwargs):
        features = self.basis_set[0].basis(X)
        for b in self.basis_set[1:]:
            features = np.concatenate((features, b.basis(X)), axis=1)
        return features

    def deriv(self, X, deriv_order=1):
        grad = self.basis_set[0].deriv(X, deriv_order=deriv_order)
        for b in self.basis_set[1:]:
            # print(grad.shape, b.deriv(X, deriv_order=deriv_order).shape)
            features = np.concatenate((grad, b.deriv(X, deriv_order=deriv_order)), axis=1)
        return features

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        raise NotImplementedError("Don't try this")
