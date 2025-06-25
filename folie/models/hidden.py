from .overdamped import Overdamped
from .underdamped import Underdamped
from ..functions.base import Function
from .._numpy import np


class OverdampedHidden(Overdamped):
    """
    A class that implements an overdamped model with some extra hidden variables linearly correlated with the visible ones.


    d \\begin{pmatrix} X(t) \\ h(t) \\end{pmatrix} = f(X,t)dt+ gamma(X,t)h(t)dt + sigma(X,t)dW_t

    where

    f(X,t) is a $dim_x + dim_h$ vector

    gamma(X,t) is a $(dim_x + dim_h) \times dim_h$ matrix

    sigma(X,t) is a $(dim_x + dim_h) \times (dim_x + dim_h)$ matrix

    """

    def __init__(self, pos_drift, friction, diffusion, dim=1, dim_h=0, **kwargs):
        self.dim_h = dim_h
        self.dim_x = dim
        pos_drift.dim_x = self.dim_x
        diffusion.dim_x = self.dim_x
        friction.dim_x = self.dim_x

        super().__init__(pos_drift, diffusion, dim=self.dim_x + self.dim_h, **kwargs)
        self.friction = friction.resize((self.dim, self.dim_h))

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self.pos_drift.coefficients.ravel(), self.friction.coefficients.ravel(), self.diffusion.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals.ravel()[: self.pos_drift.size]
        self.friction.coefficients = vals.ravel()[self.pos_drift.size : self.pos_drift.size + self.friction.size]
        self.diffusion.coefficients = vals.ravel()[self.pos_drift.size + self.friction.size :]

    @property
    def coefficients_friction(self):
        return self.friction.coefficients

    @coefficients_friction.setter
    def coefficients_friction(self, vals):
        self.friction.coefficients = vals

    def _drift(self, x, *args, **kwargs):
        return self.pos_drift(x, *args, **kwargs) + np.einsum("t...h,th-> t...", self.friction(x, *args, **kwargs), x[:, self.dim_x :])

    def _drift_dx(self, x, *args, **kwargs):
        return self.pos_drift.grad_x(x, *args, **kwargs) + np.einsum("t...he,th-> t...e", self.friction.grad_x(x, *args, **kwargs), x[:, self.dim_x :])

    def _drift_d2x(self, x, *args, **kwargs):
        return self.pos_drift.hessian_x(x, *args, **kwargs) + np.einsum("t...hef,th-> t...ef", self.friction.hessian_x(x, *args, **kwargs), x[:, self.dim_x :])

    def _drift_dcoeffs(self, x, *args, **kwargs):
        """
        Jacobian of the drift with respect to coefficients
        """
        return np.concatenate((self.pos_drift.grad_coeffs(x, *args, **kwargs), np.einsum("t...hc,th-> t...c", self.friction.grad_coeffs(x, *args, **kwargs), x[:, self.dim_x :])), axis=-1)

    @property
    def coefficients_drift(self):
        """Access the coefficients"""
        return np.concatenate((self.pos_drift.coefficients.ravel(), self.friction.coefficients.ravel()))

    @coefficients_drift.setter
    def coefficients_drift(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.pos_drift.coefficients = vals.ravel()[: self.pos_drift.size]
        self.friction.coefficients = vals.ravel()[self.pos_drift.size : self.pos_drift.size + self.friction.size]


class UnderdampedHidden(Underdamped):
    """
    A class that implements an underdamped model with some extra hidden variables linearly correlated with the visible ones.

    
    d \\begin{pmatrix} X(t) \\ H(t) \\end{pmatrix} = \\begin{pmatrix} v(t)dt \\ f(X,t)dt - gamma(X,t)H(t)dt + sigma(X,t)dW_t \\end{pmatrix}

    where

    H(t) = \\begin{pmatrix} v(t) \\ h(t) \\end{pmatrix} is a $dim_h$ vector.

    f(X,t) = \\begin{pmatrix} f(X,t) \\ 0 \\end{pmatrix} is a $dim_x + dim_h$ vector

    gamma(X,t) is a $dim_h \times dim_h$ matrix

    sigma(X,t) is a $dim_h \times dim_h$ matrix

    """

    def __init__(self, pos_drift, friction, diffusion, dim=1, dim_h=0, **kwargs):
        self.dim_h = dim_h
        self.dim_x = dim
        pos_drift.dim_x = self.dim_x
        if self.dim_h > 0:
            pos_drift = _ZeroPadHidden(pos_drift, dim_h=self.dim_h)
        diffusion.dim_x = self.dim_x
        friction.dim_x = self.dim_x
        super().__init__(pos_drift, friction, diffusion, dim=self.dim_x + self.dim_h, **kwargs)
                
    @property
    def coefficients_friction(self):
        return self.friction.coefficients

    @coefficients_friction.setter
    def coefficients_friction(self, vals):
        self.friction.coefficients = vals

class _ZeroPadHidden(Function):
    """
    Wrap a visible‐only Function f(x) and make it live in R^(dim_x+dim_h),
    by appending dim_h zeros to every output and every derivative.
    """
    def __init__(self, f_visible: Function, dim_h: int):
        self._f = f_visible
        self.dim_h = dim_h
        
        out_shape = (f_visible.dim_x + dim_h,)
        super().__init__(f_visible.domain, out_shape)

    def __repr__(self):
        return self._f.__repr__()

    @property
    def coefficients(self):
        return self._f.coefficients
        
    @coefficients.setter
    def coefficients(self, c):
        self._f.coefficients = c

    @property
    def size(self):
        return self._f.size

    def transform(self, x, *a, **kw):
        fx = self._f.transform(x, *a, **kw)
        zeros = np.zeros((fx.shape[0], self.dim_h))
        return np.concatenate((fx, zeros), axis=1)

    def transform_dx(self, x, *a, **kw):
        G = self._f.transform_dx(x, *a, **kw)   # (n, dim_x, x_dim)
        pad = np.zeros((G.shape[0], self.dim_h, x.shape[1]))
        return np.concatenate((G, pad), axis=1)

    def transform_d2x(self, x, *a, **kw):
        H = self._f.transform_d2x(x, *a, **kw)  # (n, dim_x, x_dim, x_dim)
        pad = np.zeros((H.shape[0], self.dim_h, x.shape[1], x.shape[1]))
        return np.concatenate((H, pad), axis=1)

    def transform_dcoeffs(self, x, *a, **kw):
        J = self._f.transform_dcoeffs(x, *a, **kw)  # (n, dim_x, n_coeff)
        pad = np.zeros((J.shape[0], self.dim_h, self.size))
        return np.concatenate((J, pad), axis=1)

        
