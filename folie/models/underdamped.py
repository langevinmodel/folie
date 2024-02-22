import numpy as np

from .overdamped import Overdamped


class Underdamped(Overdamped):
    def __init__(self, force, friction, diffusion, dim=1, **kwargs):
        """
        Base model for underdamped Langevin equations, defined by

        dX(t) = V(t)

        dV(t) = f(X,t)dt+ gamma(X,t)V(t)dt + sigma(X,t)dW_t

        """
        super().__init__(force, diffusion, dim=dim)
        self._friction = friction.resize((self.dim, self.dim))  # TODO: A changer pour gérer le cas 1D
        self.coefficients = np.concatenate((np.zeros(self.force.size), np.ones(self.diffusion.size), np.ones(self.friction.size)))

    @property
    def dim(self):
        """
        Dimensionnality of the model
        """
        return self._dim

    @dim.setter
    def dim(self, dim):
        self._dim = dim
        if dim >= 1:
            force_shape = (dim,)
            diffusion_shape = (dim, dim)
        else:
            force_shape = ()
            diffusion_shape = ()
        self._force = self._force.resize(force_shape)
        self._diffusion = self._diffusion.resize(diffusion_shape)
        self._friction = self._friction.resize(diffusion_shape)

    def meandispl(self, x, v, t: float = 0.0):
        return self._force(x[:, : self.dim_x]) + np.einsum("tdh,th-> td", self.friction(x[:, : self.dim_x]), v)

    def meandispl_x(self, x, v, t: float = 0.0):
        return self._force.grad_x(x[:, : self.dim_x]) + np.einsum("tdhe,th-> tde", self._friction.grad_x(x[:, : self.dim_x]), v)

    def meandispl_jac_coeffs(self, x, v, t: float = 0.0):
        """
        Jacobian of the force with respect to coefficients
        """
        return np.concatenate((self._force.grad_coeffs(x[:, : self.dim_x]), np.einsum("tdhc,th-> tdc", self._friction.grad_coeffs(x[:, : self.dim_x]), v)), axis=-1)

    def friction(self, x, t: float = 0.0):
        return self.friction(x[:, : self.dim_x])

    def friction_x(self, x, t: float = 0.0):
        return self.friction.grad_x(x[:, : self.dim_x])

    @property
    def coefficients(self):
        """Access the coefficients"""
        return np.concatenate((self._force.coefficients.ravel(), self._diffusion.coefficients.ravel(), self._friction.coefficients.ravel()))

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._force.coefficients = vals.ravel()[: self.force.size]
        self._diffusion.coefficients = vals.ravel()[self.force.size : self.force.size + self.diffusion.size]
        self._friction.coefficients = vals.ravel()[self.force.size + self.diffusion.size :]

    @property
    def coefficients_friction(self):
        return self._force.coefficients

    @coefficients_friction.setter
    def coefficients_friction(self, vals):
        self._friction.coefficients = vals
