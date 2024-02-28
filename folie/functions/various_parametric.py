import numpy as np

from .base import ParametricFunction


class sklearnTransformer(ParametricFunction):
    """
    Take any sklearn transformer and build a fonction from it
    """

    def __init__(self, transformer, output_shape=(), coefficients=None):
        super().__init__(output_shape, coefficients)
        self.transformer = transformer

    def fit(self, X, y=None, **kwargs):
        self.transformer = self.transformer.fit(X)
        self.n_functions_features_ = self.transformer.transform(X[:5, :]).shape[1]
        super().fit(X, y, **kwargs)
        return self

    def transform(self, x, *args, **kwargs):
        return self.transformer.transform(x) @ self._coefficients

    def transform_coeffs(self, x, **kwargs):
        transform_coeffs = np.eye(self.size).reshape(self.n_functions_features_, *self.output_shape_, self.size)
        return np.tensordot(self.transformer.transform(x), transform_coeffs, axes=1)


# TODO: FreezeCoefficients
class FreezeCoefficients(ParametricFunction):
    r"""
    Function that only expose a subset of the coefficients of the underlying function
    """

    def __init__(self, f, freezed_coefficients):
        self.f = f

    def fit(self, x, y=None, **kwargs):
        self.f.fit(x, y, **kwargs)
        self.fitted_ = True
        return self

    def transform(self, x, *args, **kwargs):
        r"""Transforms the input data."""
        return self.f.transform(x)

    def transform_x(self, x, **kwargs):
        r"""Gradient of the function with respect to input data"""

        return self.f.transform_x(x)

    def hessian_x(self, x, **kwargs):
        """
        Hessian of the function with respect to input data
        """
        return self.f.hessian_x(x)

    def transform_coeffs(self, x, **kwargs):
        r"""Transforms the input data."""


class FunctionOffsetWithCoefficient(ParametricFunction):
    """
    A composition function for returning f(x)+g(x)*y
    """

    def __init__(self, f, g=None, output_shape=(), **kwargs):
        self.f = f
        self.g = g
        # Check if g is a Function or a constant
        super().__init__(self.f.output_shape_)

    def fit(self, X, y=None, **kwargs):
        # First fit sub function with representative array then fit the global one
        pass

    def transform(self, x, v, *args, **kwargs):
        return self.f(x, *args, **kwargs) + np.einsum("t...h,th-> t...", self.g(x, *args, **kwargs), v)

    def transform_x(self, x, v, *args, **kwargs):
        return self.f.transform_x(x, *args, **kwargs) + np.einsum("t...he,th-> t...e", self.g.transform_x(x, *args, **kwargs), v)

    def transform_coeffs(self, x, v, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        return np.concatenate((self._force.transform_coeffs(x, *args, **kwargs), np.einsum("t...hc,th-> t...c", self._friction.transform_coeffs(x, *args, **kwargs), v)), axis=-1)


class ModelOverlay(ParametricFunction):
    """
    A class that allow to overlay a model and make it be used as a function
    """

    def __init__(self, model, function_name, output_shape=None, **kwargs):
        self.model = model
        # Do some check
        if not hasattr(self.model, "_" + function_name):
            raise ValueError("Model does not implement " + "_" + function_name + ".")
        self.function_name = function_name
        # Define output shape from model dimension
        if output_shape is None and model.dim <= 1:
            output_shape = ()
        elif output_shape is None:
            raise ValueError("output_shape should be defined.")

        super().__init__(output_shape, coefficients=self.coefficients, **kwargs)

    def transform(self, x, *args, **kwargs):
        return getattr(self.model, "_" + self.function_name)(x, *args, **kwargs)

    def transform_x(self, x, *args, **kwargs):
        if hasattr(self.model, self.function_name + "_x"):
            return getattr(self.model, self.function_name + "_x")(x, *args, **kwargs)
        else:
            return super().transform_x(x, *args, **kwargs)  # If not implemented use finite difference

    def transform_coeffs(self, x, *args, **kwargs):
        """
        Jacobian of the force with respect to coefficients
        """
        if hasattr(self.model, self.function_name + "_jac_coeffs"):
            return getattr(self.model, self.function_name + "_jac_coeffs")(x, *args, **kwargs)
        else:
            return super().transform_coeffs(x, *args, **kwargs)  # If not implemented use finite difference

    @property
    def coefficients(self):
        """Access the coefficients"""
        return getattr(self.model, "coefficients_" + self.function_name)

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        setattr(self.model, "coefficients_", vals)
