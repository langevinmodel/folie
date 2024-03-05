"""numerical differentiation function, gradient, and Hessian

Author : josef-pkt
License : BSD

Notes
-----
These are simple forward differentiation, so that we have them available
without dependencies.

* numerical precision will vary and depend on the choice of stepsizes

Adapated from statsmodels
"""

from .._numpy import np

# NOTE: we only do double precision internally so far
EPS = np.finfo(float).eps


def _get_epsilon(x, s, epsilon, n):
    if epsilon is None:
        h = EPS ** (1.0 / s) * np.maximum(np.abs(x), 0.1)
    else:
        if np.isscalar(epsilon):
            h = np.empty(n)
            h.fill(epsilon)
        else:  # pragma : no cover
            h = np.asarray(epsilon)
            if h.shape != x.shape:
                raise ValueError("If h is not a scalar it must have the same" " shape as x.")
    return h


def approx_fprime(x, f, output_shape, epsilon=None, centered=False):
    """
    Gradient of function vectorized for scalar parameter.

    This assumes that the function ``f`` is vectorized for a scalar parameter.
    The function value ``f(x)`` has then the same shape as the input ``x``.
    The derivative returned by this function also has the same shape as ``x``.

    Parameters
    ----------
    x : ndarray
        Parameters at which the derivative is evaluated.
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.
    centered : bool
        Whether central difference should be returned. If not, does forward
        differencing.

    Returns
    -------
    grad : ndarray
        Array of derivatives, gradient evaluated at parameters ``x``.
    """
    n = x.shape[0]
    dim = x.shape[1]
    grad = np.zeros((n, *output_shape, dim), np.promote_types(float, x.dtype))
    ei = np.zeros((dim,), float)
    if not centered:
        eps = _get_epsilon(np.abs(x).max(axis=0), 2, epsilon, dim)
        f0 = f(x)
        for k in range(dim):
            ei[k] = eps[k]
            grad[..., k] = (f(x + ei) - f0) / eps[k]
            ei[k] = 0.0

    else:
        eps = _get_epsilon(np.abs(x).max(axis=0), 3, epsilon, dim) / 2.0
        for k in range(dim):
            ei[k] = eps[k]
            grad[..., k] = (f(x + ei) - f(x - ei)) / eps[k]
            ei[k] = 0.0

    return grad


_hessian_docs = """
    Calculate Hessian with finite difference derivative approximation

    Parameters
    ----------
    x : array_like
       value at which function derivative is evaluated
    f : function
       function of one array f(x, `*args`, `**kwargs`)
    epsilon : float or array_like, optional
       Stepsize used, if None, then stepsize is automatically chosen
       according to EPS**(1/%(scale)s)*x.
    args : tuple
        Arguments for function `f`.
    kwargs : dict
        Keyword arguments for function `f`.
    %(extra_params)s

    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian
    %(extra_returns)s

    Notes
    -----
    Equation (%(equation_number)s) in Ridout. Computes the Hessian as::

      %(equation)s

    where e[j] is a vector with element j == 1 and the rest are zero and
    d[i] is epsilon[i].

    References
    ----------:

    Ridout, M.S. (2009) Statistical applications of the complex-step method
        of numerical differentiation. The American Statistician, 63, 66-74
"""


def approx_hess(x, f, epsilon=None, args=(), kwargs={}):
    n = len(x)
    h = _get_epsilon(x, 4, epsilon, n)
    ee = np.diag(h)
    hess = np.outer(h, h)

    for i in range(n):
        for j in range(i, n):
            hess[i, j] = (f(*((x + ee[i, :] + ee[j, :],) + args), **kwargs) - f(*((x + ee[i, :] - ee[j, :],) + args), **kwargs) - (f(*((x - ee[i, :] + ee[j, :],) + args), **kwargs) - f(*((x - ee[i, :] - ee[j, :],) + args), **kwargs))) / (4.0 * hess[i, j])
            hess[j, i] = hess[i, j]
    return hess
