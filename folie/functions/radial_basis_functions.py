from .base import ParametricFunction
from .._numpy import np

from scipy.spatial.distance import cdist


def gaussian(r):
    return np.exp(-(r**2))


def linear(r):
    return r


def quadratic(r):
    return r**2


def inverse_quadratic(r):
    return 1.0 / (1.0 + r**2)


def multiquadric(r):
    return np.sqrt((1.0 + r**2))


def inverse_multiquadric(r):
    return 1.0 / np.sqrt(1.0 + r**2)


def spline(r):
    return r**2 * np.log(r + 1.0)


def poisson_one(r):
    return (r - 1.0) * np.exp(-r)


def poisson_two(r):
    return ((r - 2.0) / 2.0) * r * np.exp(-r)


def matern32(r):
    return (1.0 + 3**0.5 * r) * np.exp(-(3**0.5) * r)


def matern52(r):
    return (1.0 + 5**0.5 * r + (5 / 3) * r**2) * np.exp(-(5**0.5) * r)


def sigmoid(r):
    return np.tanh(r)


bases = {
    "gaussian": gaussian,
    "linear": linear,
    "quadratic": quadratic,
    "inverse quadratic": inverse_quadratic,
    "multiquadric": multiquadric,
    "inverse multiquadric": inverse_multiquadric,
    "spline": spline,
    "poisson one": poisson_one,
    "poisson two": poisson_two,
    "matern32": matern32,
    "matern52": matern52,
    "sigmoid": sigmoid,
}


class RadialBasisFunction(ParametricFunction):
    def __init__(self, domain, sigma="from_grid", rbf="gaussian", output_shape=(), coefficients=None):
        if isinstance(rbf, str):
            self.rbf = bases[rbf]
        elif callable(rbf):
            self.rbf = rbf
        else:
            raise ValueError("rbf value should be either a string or a callable")
        self.ref_X = domain.mesh.p.T

        self.n_functions_features_ = self.ref_X.shape[0]

        if sigma == "from_grid":

            locals_dists = np.sort(cdist(self.ref_X, self.ref_X), axis=1)[:, 1 : 1 + 2**domain.dim]
            self.sigmas = 1.0 / locals_dists.mean(axis=1)
        else:
            self.sigmas = sigma

        super().__init__(domain, output_shape, coefficients)

    def transform(self, X, *args, **kwargs):
        r = cdist(X, self.ref_X)

        return (self._coefficients[None, ...] * self.rbf(r * self.sigmas)[..., None]).sum(axis=1)

    def transform_dcoeffs(self, X, *args, **kwargs):
        transform_dcoeffs = np.eye(self.size).reshape(self.n_functions_features_, -1)
        r = cdist(X, self.ref_X)
        return (transform_dcoeffs[None, ...] * self.rbf(r * self.sigmas)[..., None]).sum(axis=1)
