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

from ._numpy import np

from scipy.optimize import approx_fprime

# NOTE: we only do double precision internally so far
EPS = np.finfo(float).eps


def value_and_grad(fun, argnums=0, eps=1.4901161193847657e-08):
    """Returns a function that returns both value and gradient. Suitable for use
    in scipy.optimize"""
    def val_grad_fn(x,*args,**kwargs):
        return fun(x,*args,**kwargs), jacobian(fun,argnums, eps)(x,*args,**kwargs)
    return val_grad_fn



def jacobian(fun, argnums=0, eps=1.4901161193847657e-08):

    def grad_fn(*args,**kwargs):
        def fun_wrapper(arg):
            args_list = list(args)
            args_list[argnums] = arg
            return fun(*args_list,**kwargs)
        
        arg = np.array(args[argnums], dtype=float)
        grad = approx_fprime(arg, fun_wrapper, eps)
        return grad
    
    return grad_fn





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




def grad_x(f,x,*args, epsilon=None, centered=False, **kwargs):
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
    f0 = f(x,*args, **kwargs)
    output_shape = f0.shape[1:]
    grad = np.zeros((n, *output_shape, dim), np.promote_types(float, x.dtype))
    ei = np.zeros((dim,), float)
    if not centered:
        eps = _get_epsilon(np.abs(x).max(axis=0), 2, epsilon, dim)
        for k in range(dim):
            ei[k] = eps[k]
            grad[..., k] = (f(x + ei,*args, **kwargs) - f0) / eps[k]
            ei[k] = 0.0

    else:
        eps = _get_epsilon(np.abs(x).max(axis=0), 3, epsilon, dim) / 2.0
        for k in range(dim):
            ei[k] = eps[k]
            grad[..., k] = (f(x + ei,*args, **kwargs) - f(x - ei,*args, **kwargs)) / eps[k]
            ei[k] = 0.0

    return grad



def hessian_x(f,x, *args,**kwargs):
    return grad_x(lambda x, *args,**kwargs: grad_x(f,x, *args, epsilon=1e-4,**kwargs) ,x, *args, epsilon=1e-4,**kwargs)