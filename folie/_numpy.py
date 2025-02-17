import os

_which_numpy = os.environ.get("FOLIE_NUMPY", "numpy").lower()



if _which_numpy == "jax":
    import jax.numpy as np
    from jax import grad,value_and_grad

elif _which_numpy == "autograd":
    import autograd.numpy as np
    from autograd import elementwise_grad
    from autograd import value_and_grad, jacobian

    def grad_x(f,x,*args,**kwargs):
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
        Returns
        -------
        grad : ndarray
            Array of derivatives, gradient evaluated at parameters ``x``.
        """
        n = x.shape[0]
        dim = x.shape[1]
        f0 = f(x)
        output_shape = f0.shape[1:]
        grad = np.zeros((n, *output_shape, dim), np.promote_types(float, x.dtype))
        for n in range(dim):
            grad[...,n]= elementwise_grad(lambda x:f(x,*args,**kwargs)[n])(x)
        return grad
    
    def hessian_x(f,x,*args,**kwargs):
        n = x.shape[0]
        dim = x.shape[1]
        f0 = f(x)
        output_shape = f0.shape[1:]
        hessian = np.zeros((n, *output_shape, dim,dim), np.promote_types(float, x.dtype))
        for n in range(dim):
            for m in range(dim):
                hessian[...,n]= elementwise_grad(elementwise_grad(lambda x:f(x,*args,**kwargs)[n,m]))(x)
        return hessian

else:
    import numpy as np
    from ._numdifference import grad_x,value_and_grad,jacobian, hessian_x  # TODO: adapt the numerical differentiation to be called in a similar fashion


