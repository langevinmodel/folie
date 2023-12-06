"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

import numpy as np
from .transitionDensity import TransitionDensity


class ExactDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the exact transition density for a model (when available)
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity(self, x0, xt, t0, dt: float):
        """
        The exact transition density (when applicable)
        Note: this will raise exception if the model does not implement exact_density
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        return self._model.exact_density(x0=x0, xt=xt, t0=t0, dt=dt)

    def run_step(self, x, dt, dW):
        return self._model.exact_step(x, dt, dW)


class EulerDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity(self, x0, xt, t0, dt: float):
        """
        The transition density obtained via Euler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig2t = (self._model.diffusion(x0, t0).ravel()) * 2 * dt
        mut = x0.ravel() + self._model.force(x0, t0).ravel() * dt
        return -((xt.ravel() - mut) ** 2) / sig2t - 0.5 * np.log(np.pi * sig2t)

    def run_step(self, x, dt, dW, t=0.0):
        sig_sq_dt = np.sqrt(self._model.diffusion(x, t) * dt)
        return x + self._model.force(x, t) * dt + sig_sq_dt * dW


class OzakiDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Ozaki approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity(self, x0, xt, t0, dt: float):
        """
        The transition density obtained via Ozaki expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0).ravel()
        mu = self._model.force(x0, t0).ravel()
        mu_x = self._model.force_x(x0, t0).ravel()
        temp = mu * (np.exp(mu_x * dt) - 1) / mu_x

        Mt = x0.ravel() + temp
        Kt = (2 / dt) * np.log(1 + temp / x0.ravel())
        Vt = np.sqrt(sig * (np.exp(Kt * dt) - 1) / Kt)

        return -0.5 * ((xt.ravel() - Mt) / Vt) ** 2 - 0.5 * np.log(2 * np.pi) - np.log(Vt)


class ShojiOzakiDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Shoji-Ozaki approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity(self, x0, xt, t0, dt: float):
        """
        The transition density obtained via Shoji-Ozaki expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = np.sqrt(self._model.diffusion(x0, t0).ravel())
        mu = self._model.force(x0, t0).ravel()

        Mt = 0.5 * sig**2 * self._model.force_xx(x0, t0).ravel() + self._model.force_t(x0, t0)
        Lt = self._model.force_x(x0, t0).ravel()
        if (Lt == 0).any():  # TODO: need to fix this
            B = sig * np.sqrt(dt)
            A = x0.ravel() + mu * dt + Mt * dt**2 / 2
        else:
            B = sig * np.sqrt((np.exp(2 * Lt * dt) - 1) / (2 * Lt))

            elt = np.exp(Lt * dt) - 1
            A = x0.ravel() + mu / Lt * elt + Mt / (Lt**2) * (elt - Lt * dt)

        return -0.5 * ((xt.ravel() - A) / B) ** 2 - 0.5 * np.log(2 * np.pi) - np.log(B)


class ElerianDensity(EulerDensity):
    def __init__(self, model):
        """
        Class which represents the Elerian (Milstein) approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity(self, x0, xt, t0, dt: float):
        """
        The transition density obtained via Milstein Expansion (Elarian density).
        When d(sigma)/dx = 0, reduces to Euler
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param t0: float, the time of at which to evalate the coefficients. Irrelevant For time inhomogenous models
        :param dt: float, the time step between x0 and xt
        :return: probability (same dimension as x0 and xt)
        """
        sig_x = self._model.diffusion_x(x0, t0).ravel()
        if isinstance(x0, np.ndarray) and (sig_x == 0).any:
            return super()._logdensity(x0=x0, xt=xt, t0=t0, dt=dt)

        sig = self._model.diffusion(x0, t0).ravel()
        mu = self._model.force(x0, t0).ravel()

        A = sig * sig_x * dt * 0.5
        B = -0.5 * sig / sig_x + x0.ravel() + mu * dt - A
        z = (xt.ravel() - B) / A
        C = 1.0 / (sig_x**2 * dt)

        scz = np.sqrt(C * z)
        cpz = -0.5 * (C + z)
        ch = np.exp(scz + cpz) + np.exp(-scz + cpz)
        return -0.5 * np.log(z) + np.log(ch) - np.log(2 * np.abs(A) * np.sqrt(2 * np.pi))


class KesslerDensity(EulerDensity):
    def __init__(self, model):
        """
        Class which represents the Kessler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity(self, x0, xt, t0, dt: float):
        """
        The transition density obtained via Kessler expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0).ravel()
        sig_x = self._model.diffusion_x(x0, t0).ravel()
        sig_xx = self._model.diffusion_xx(x0, t0).ravel()
        mu = self._model.force(x0, t0).ravel()
        mu_x = self._model.force_x(x0, t0).ravel()
        mu_xx = self._model.force_xx(x0, t0).ravel()
        x0 = x0.ravel()
        d = dt**2 / 2
        E = x0 + mu * dt + (mu * mu_x + 0.5 * sig * mu_xx) * d

        V = x0**2 + (2 * mu * x0 + sig) * dt + (2 * mu * (mu_x * x0 + mu + 0.5 * sig_x) + sig * (mu_xx * x0 + 2 * mu_x + 0.5 * sig_xx)) * d - E**2
        V = np.abs(V)
        return -0.5 * ((xt.ravel() - E) ** 2 / V) - 0.5 * np.log(np.sqrt(2 * np.pi) * V)


class DrozdovDensity(EulerDensity):
    def __init__(self, model):
        """
        Class which represents the Drozdov approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity(self, x0, xt, t0, dt: float):
        """
        The transition density obtained via Drozdov expansion
        :param x0: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x0)
        :param dt: float, the time of observing Xt
        :return: probability (same dimension as x0 and xt)
        """
        sig = self._model.diffusion(x0, t0).ravel()
        sig_x = self._model.diffusion_x(x0, t0).ravel()
        sig_xx = self._model.diffusion_xx(x0, t0).ravel()
        mu = self._model.force(x0, t0).ravel()
        mu_x = self._model.force_x(x0, t0).ravel()
        mu_xx = self._model.force_xx(x0, t0).ravel()

        d = dt**2 / 2
        E = x0.ravel() + mu * dt + (mu * mu_x + 0.5 * sig * mu_xx) * d

        V = sig * dt + (mu * sig_x + 2 * mu_x * sig + sig * sig_xx) * d
        V = np.abs(V)
        return -0.5 * ((xt.ravel() - E) ** 2 / V) - 0.5 * np.log(np.sqrt(2 * np.pi) * V)
