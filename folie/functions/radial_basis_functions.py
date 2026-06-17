from .base import ParametricFunction
from .._numpy import np

from scipy.spatial.distance import cdist


def gaussian(r):
    r"""Gaussian radial basis: :math:`\phi(r) = e^{-r^2}`."""
    return np.exp(-(r**2))


def linear(r):
    r"""Linear radial basis: :math:`\phi(r) = r`."""
    return r


def quadratic(r):
    r"""Quadratic radial basis: :math:`\phi(r) = r^2`."""
    return r**2


def inverse_quadratic(r):
    r"""Inverse quadratic radial basis: :math:`\phi(r) = \frac{1}{1 + r^2}`."""
    return 1.0 / (1.0 + r**2)


def multiquadric(r):
    r"""Multiquadric radial basis: :math:`\phi(r) = \sqrt{1 + r^2}`."""
    return np.sqrt((1.0 + r**2))


def inverse_multiquadric(r):
    r"""Inverse multiquadric radial basis: :math:`\phi(r) = \frac{1}{\sqrt{1 + r^2}}`."""
    return 1.0 / np.sqrt(1.0 + r**2)


def spline(r):
    r"""Spline radial basis: :math:`\phi(r) = r^2 \log(r + 1)`."""
    return r**2 * np.log(r + 1.0)


def poisson_one(r):
    r"""First Poisson kernel: :math:`\phi(r) = (r - 1) e^{-r}`."""
    return (r - 1.0) * np.exp(-r)


def poisson_two(r):
    r"""Second Poisson kernel: :math:`\phi(r) = \frac{r(r - 2)}{2} e^{-r}`."""
    return ((r - 2.0) / 2.0) * r * np.exp(-r)


def matern32(r):
    r"""Matern 3/2 kernel: :math:`\phi(r) = (1 + \sqrt{3}r)e^{-\sqrt{3}r}`."""
    return (1.0 + 3**0.5 * r) * np.exp(-(3**0.5) * r)


def matern52(r):
    r"""Matern 5/2 kernel: :math:`\phi(r) = (1 + \sqrt{5}r + \frac{5}{3}r^2)e^{-\sqrt{5}r}`."""
    return (1.0 + 5**0.5 * r + (5 / 3) * r**2) * np.exp(-(5**0.5) * r)


def sigmoid(r):
    r"""Sigmoid radial basis: :math:`\phi(r) = \tanh(r)`."""
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
    r"""Radial basis function interpolation.

    Represents a function as a weighted sum of radial basis functions centered
    at reference points:

    .. math::

       f(x) = \sum_{i=1}^{N} c_i \phi\left(\frac{\|x - x_i\|}{\sigma_i}\right)

    where :math:`\phi` is a radial kernel, :math:`x_i` are reference points
    from the domain mesh, and :math:`\sigma_i` are scale parameters computed
    from the distances to nearby reference points.

    Parameters
    ----------
    domain : Domain
        The spatial domain. The mesh points are used as RBF centers.
    sigma : float, "from_grid", or array-like
        Scale parameter(s). If "from_grid", computed from nearest-neighbor
        distances of the reference grid.
    rbf : str or callable
        Radial basis kernel. Either a string key from `bases` dict or a
        callable :math:`\phi(r)`.
    output_shape : tuple, optional
        The output shape of the function. Defaults to scalar.
    coefficients : array-like, optional
        Initial coefficient values. Random values are used if not provided.

    Examples
    --------
    >>> from folie.functions import RadialBasisFunction
    >>> from folie.domains import Domain
    >>> domain = Domain.Rd(dim=1)
    >>> f = RadialBasisFunction(domain, rbf="gaussian")
    >>> f.fit(x_train, y_train)
    >>> f(x_test)

    Notes
    -----
    Number of free parameters equals the number of reference points in the domain mesh.
    Suitable for irregularly spaced data and multi-dimensional problems.
    """

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
