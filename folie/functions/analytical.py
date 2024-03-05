# A set of analytical functions for examples
from .base import ParametricFunction
from .._numpy import np


class PotentialFunction(ParametricFunction):
    dim = 1

    def __init__(self):
        if self.dim == 1:
            output_shape = ()
        else:
            output_shape = (self.dim,)
        super().__init__(output_shape, self._coefficients)

    def resize(self, new_shape):
        # We should run some test here
        pass


class DoubleWell(PotentialFunction):
    dim = 2

    def __init__(self, A, b):
        super().__init__()

    def potential(self, x, *args, **kwargs):
        return x

    def transform(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        return np.trace(self.bspline(x), axis1=1, axis2=2)


class MullerBrown(PotentialFunction):
    pass


class EntropicSwitch(PotentialFunction):
    dim = 2

    def __init__(self, A, b):
        super().__init__()

    def potential(self, x, *args, **kwargs):
        return x

    def transform(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        return np.trace(self.bspline(x), axis1=1, axis2=2)


class LogExpPotential(PotentialFunction):
    dim = 2

    def __init__(self, A, b):
        super().__init__()

    def potential(self, x, *args, **kwargs):
        return x

    def transform(self, x, *args, **kwargs):
        nsamples, dim = x.shape
        return np.trace(self.bspline(x), axis1=1, axis2=2)
