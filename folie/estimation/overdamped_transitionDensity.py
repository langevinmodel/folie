"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from .._numpy import np, grad_x, hessian_x
import warnings
from .transitionDensity import TransitionDensity
from .transitionDensity import gaussian_likelihood_1D, gaussian_likelihood_ND

try:
    from ._filter_smoother import filtersmoother
except ImportError as err:
    # print(err)
    # warnings.warn("Python fallback will been used for filtersmoother module. Consider compiling the fortran module")
    from ._kalman_python import filtersmoother


class ExactDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the exact transition density for a model (when available)

        Parameters
        ----------
            model: the SDE model, referenced during calls to the transition density
        """
        self._model = model
        self._min_prob = np.log(1e-30)  # used to floor probabilities when evaluating the log

    def _logdensity(self, x, xt, dt: float):
        """
        The exact transition density (when applicable)
        Note: this will raise exception if the model does not implement exact_density
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """
        return self._model.exact_density(x=x, xt=xt, dt=dt)


class EulerDensity(TransitionDensity):

    def __init__(self, model):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity1D(self, x, xt, dt: float, bias=0.0, **kwargs):
        """
        The transition density obtained via Euler expansion
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """
        sig2t = 2 * (self._model.diffusion(x, **kwargs)).ravel() * dt
        mut = x.ravel() + self._model.drift(x, bias, **kwargs).ravel() * dt
        return gaussian_likelihood_1D(xt, mut, sig2t)


    def _logdensityND(self, x, xt, dt, bias=0.0, **kwargs):
        """
        The transition density evaluated at these arguments
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """

        # TODO: Add correction terms
        E = x + self._model.drift(x, bias, **kwargs) * dt
        V = 2 * (self._model.diffusion(x, **kwargs)) * dt
        return gaussian_likelihood_ND(xt, E, V)

    def _hiddenvariance(self, x, xt, sigh, dt, **kwargs):
        """
        The transition density evaluated at these arguments
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """

        E2 = self._model.friction(x[:, : self._model.dim_x], **kwargs) * dt
        V = 2 * (self._model.diffusion(x[:, : self._model.dim_x], **kwargs)) * dt
        invV = np.linalg.inv(V)
        dh = self._model.dim_h
        dhdh = sigh[:, :dh, :dh] - sigh[:, :dh, dh:] - sigh[:, dh:, :dh] + sigh[:, dh:, dh:]
        hdh = sigh[:, dh:, :dh] - sigh[:, dh:, dh:]
        dhh = sigh[:, :dh, dh:] - sigh[:, dh:, dh:]
        hh = sigh[:, dh:, dh:]

        EVE = np.einsum("tdh,tdf,tfg-> thg", E2, invV, E2)
        EV = np.einsum("tdh,tdf-> thf", E2, invV)
        VE = np.einsum("tdf,tfh-> tdh", invV, E2)

        return -0.5 * np.einsum("tij,tji->t", invV[:, dh:, dh:], dhdh) + 0.5 * np.einsum("tij,tji->t", EV[:, :, dh:], dhh) + 0.5 * np.einsum("tij,tji->t", VE[:, dh:, :], hdh) - 0.5 * np.einsum("tij,tji->t", EVE, hh)

    def hiddencorrection(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        return self._hiddenvariance(x=trj["x"], xt=trj["xt"], sigh=trj["sig_h"], dt=trj["dt"]).sum()
        # return like.sum() / weight, -np.hstack((np.zeros(self._model.pos_drift.size), jac.sum(axis=0) / weight))

    def e_step(self, weight, trj, coefficients, mu0, sig0):
        """
        In presence of hidden variables, reconstruct then using a Kalman Filter.
        Assume that the model is an OverdampedHidden model
        """
        self._model.coefficients = coefficients
        muh, Sigh = filtersmoother(
            trj["xt"][:, : self._model.dim_x],
            self._model.pos_drift(trj["x"][:, : self._model.dim_x], trj["bias"][:, : self._model.dim_x]) * trj["dt"],
            self._model.friction(trj["x"][:, : self._model.dim_x]) * trj["dt"],
            2 * self._model.diffusion(trj["x"][:, : self._model.dim_x]) * trj["dt"],
            mu0,
            sig0,
        )

        trj["sig_h"] = Sigh
        trj["x"][:, self._model.dim_x :] = muh[:, self._model.dim_h :]
        trj["xt"][:, self._model.dim_x :] = muh[:, : self._model.dim_h]
        return muh[0, self._model.dim_h :] / weight, Sigh[0, self._model.dim_h :, self._model.dim_h :] / weight  # Return µ0 and sig0


class ElerianDensity(EulerDensity):
    use_jac = False

    def __init__(self, model):
        """
        Class which represents the Elerian (Milstein) approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity1D(self, x, xt, dt: float, bias=0.0, **kwargs):
        """
        The transition density obtained via Milstein Expansion (Elarian density).
        When d(sigma)/dx = 0, reduces to Euler
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """
        sig_x = 2 * grad_x(self._model.diffusion,x, **kwargs).ravel()
        if isinstance(x, np.ndarray) and (sig_x == 0).any:
            return super()._logdensity1D(x=x, xt=xt, dt=dt, bias=bias, **kwargs)[0]

        sig = 2 * self._model.diffusion(x, **kwargs).ravel()
        mu = self._model.drift(x, bias, **kwargs).ravel()

        A = sig * sig_x * dt * 0.5
        B = -0.5 * sig / sig_x + x.ravel() + mu * dt - A
        z = (xt.ravel() - B) / A
        C = 1.0 / (sig_x**2 * dt)

        scz = np.sqrt(C * z)
        cpz = -0.5 * (C + z)
        ch = np.exp(scz + cpz) + np.exp(-scz + cpz)
        return -0.5 * np.log(z) + np.log(ch) - np.log(2 * np.abs(A) * np.sqrt(2 * np.pi))


class KesslerDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Kessler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity1D(self, x, xt, dt: float, bias=0.0, **kwargs):
        """
        The transition density obtained via Kessler expansion
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x and xt)
        """
        sig = 2 * self._model.diffusion(x, **kwargs).ravel()
        sig_x = 2 * grad_x(self._model.diffusion, x, **kwargs).ravel()
        sig_xx = 2 * hessian_x(self._model.diffusion,x, **kwargs).ravel()
        mu = self._model.drift(x, bias, **kwargs).ravel()
        mu_x = grad_x(self._model.drift,x, bias, **kwargs).ravel()
        mu_xx = hessian_x(self._model.drift,x, bias, **kwargs).ravel()
        x = x.ravel()
        d = dt**2 / 2
        E = x + mu * dt + (mu * mu_x + 0.5 * sig * mu_xx) * d

        V = x**2 + (2 * mu * x + sig) * dt + (2 * mu * (mu_x * x + mu + 0.5 * sig_x) + sig * (mu_xx * x + 2 * mu_x + 0.5 * sig_xx)) * d - E**2
        V = np.abs(V)
        return gaussian_likelihood_1D(xt, E, V)


class DrozdovDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Drozdov approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity1D(self, x, xt, dt: float, bias=0.0, **kwargs):
        """
        The transition density obtained via Drozdov expansion
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x and xt)
        """
        sig = 2 * self._model.diffusion(x, **kwargs).ravel()
        sig_x = 2 * grad_x(self._model.diffusion, x, **kwargs).ravel()
        sig_xx = 2 * hessian_x(self._model.diffusion,x, **kwargs).ravel()
        mu = self._model.drift(x, bias, **kwargs).ravel()
        mu_x = grad_x(self._model.drift,x, bias, **kwargs).ravel()
        mu_xx = hessian_x(self._model.drift,x, bias, **kwargs).ravel()

        d = dt**2 / 2
        E = x.ravel() + mu * dt + (mu * mu_x + 0.5 * sig * mu_xx) * d

        V = sig * dt + (mu * sig_x + 2 * mu_x * sig + sig * sig_xx) * d
        V = np.abs(V)
        return gaussian_likelihood_1D(xt, E, V)
