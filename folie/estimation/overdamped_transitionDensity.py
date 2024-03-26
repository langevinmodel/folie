"""
The code in this file is copied and adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from .._numpy import np
import warnings
from .transitionDensity import TransitionDensity

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
    use_jac = True

    def __init__(self, model):
        """
        Class which represents the Euler approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def __call__(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        like, jac = self._logdensity(**trj)
        return (np.asarray(-np.sum(np.maximum(self._min_prob, like)) / weight), np.asarray(-np.sum(jac, axis=0) / weight))

    def _logdensity1D(self, x, xt, dt: float, bias=0.0, **kwargs):
        """
        The transition density obtained via Euler expansion
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """
        sig2t = (self._model.diffusion(x, **kwargs)).ravel() * dt
        mut = x.ravel() + self._model.meandispl(x, bias, **kwargs).ravel() * dt
        ll = self._gaussian_likelihood_1D(xt, mut, sig2t)
        if not self.use_jac:
            return ll, np.zeros(2)

        jacV = (self._model.diffusion.grad_coeffs(x, **kwargs)) * 2 * dt
        l_jac_mu = 2 * ((xt.ravel() - mut) / sig2t)[:, None] * self._model.meandispl.grad_coeffs(x, bias, **kwargs) * dt
        l_jac_V = (((xt.ravel() - mut) ** 2) / sig2t ** 2)[:, None] * jacV - 0.5 * jacV / sig2t[:, None]
        return ll, np.concatenate((l_jac_mu, l_jac_V), axis=-1)

    def _logdensityND(self, x, xt, dt, bias=0.0, **kwargs):
        """
        The transition density evaluated at these arguments
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """

        # TODO: Add correction terms
        E = x + self._model.meandispl(x, bias, **kwargs) * dt
        V = (self._model.diffusion(x, **kwargs)) * dt
        invVE = np.linalg.solve(V, xt - E)
        ll = -0.5 * np.einsum("ti,ti-> t", invVE, xt - E) - 0.5 * np.log(np.sqrt(2 * np.pi) * np.linalg.det(V))

        if not self.use_jac:
            return ll, np.zeros(2)

        jacV = (self._model.diffusion.grad_coeffs(x, **kwargs)) * dt
        invV = np.linalg.inv(V)  # TODO: Use linalg.solve instead of inv ?
        l_jac_E = np.einsum("ti,tic-> tc", invVE, self._model.meandispl.grad_coeffs(x, bias, **kwargs) * dt)
        l_jac_V = 0.5 * np.einsum("ti,tijc,tj-> tc", xt - E, np.einsum("tij,tjkc,tkl->tilc", invV, jacV, invV), xt - E) - 0.5 * np.einsum("tijc,tji->tc", jacV, invV)
        return ll, np.concatenate((l_jac_E, l_jac_V), axis=-1)

    def _hiddenvariance(self, x, xt, sigh, dt, **kwargs):
        """
        The transition density evaluated at these arguments
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """

        E2 = self._model.friction(x[:, : self._model.dim_x], **kwargs) * dt
        V = (self._model.diffusion(x[:, : self._model.dim_x], **kwargs)) * dt
        invV = np.linalg.inv(V)
        dh = self._model.dim_h
        dhdh = sigh[:, :dh, :dh] - sigh[:, :dh, dh:] - sigh[:, dh:, :dh] + sigh[:, dh:, dh:]
        hdh = sigh[:, dh:, :dh] - sigh[:, dh:, dh:]
        dhh = sigh[:, :dh, dh:] - sigh[:, dh:, dh:]
        hh = sigh[:, dh:, dh:]

        EVE = np.einsum("tdh,tdf,tfg-> thg", E2, invV, E2)
        EV = np.einsum("tdh,tdf-> thf", E2, invV)
        VE = np.einsum("tdf,tfh-> tdh", invV, E2)

        extra_ll = -0.5 * np.einsum("tij,tji->t", invV[:, dh:, dh:], dhdh) + 0.5 * np.einsum("tij,tji->t", EV[:, :, dh:], dhh) + 0.5 * np.einsum("tij,tji->t", VE[:, dh:, :], hdh) - 0.5 * np.einsum("tij,tji->t", EVE, hh)
        jacE2 = self._model.friction.grad_coeffs(x[:, : self._model.dim_x], **kwargs) * dt

        EVjE = np.einsum("tdh,tdf,tfgc-> thgc", E2, invV, jacE2)
        jEV = np.einsum("tdhc,tdf-> thfc", jacE2, invV)
        VjE = np.einsum("tdf,tfhc-> tdhc", invV, jacE2)

        l_jac_E = -0.5 * np.einsum("tijc,tji->tc", jEV[:, :, dh:, :], dhh) - 0.5 * np.einsum("tijc,tji->tc", VjE[:, dh:, ...], hdh) + np.einsum("tijc,tji->tc", EVjE, hh)

        jacV = self._model.diffusion.grad_coeffs(x[:, : self._model.dim_x], **kwargs) * dt
        jacinvV = -np.einsum("tij,tjkc,tkl->tilc", invV, jacV, invV)
        EjVE = np.einsum("tdh,tdfc,tfg-> thgc", E2, jacinvV, E2)
        EjV = np.einsum("tdh,tdfc-> thfc", E2, jacinvV)
        jVE = np.einsum("tdfc,tfh-> tdhc", jacinvV, E2)
        l_jac_V = 0.5 * np.einsum("tijc,tji->tc", jacinvV[:, dh:, dh:, :], dhdh) - 0.5 * np.einsum("tijc,tji->tc", jVE[:, dh:, :], hdh) - 0.5 * np.einsum("tijc,tji->tc", EjV[:, :, dh:], dhh) + 0.5 * np.einsum("tijc,tji->tc", EjVE, hh)
        return extra_ll, np.concatenate((l_jac_E, l_jac_V), axis=-1)

    def hiddencorrection(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        like, jac = self._hiddenvariance(x=trj["x"], xt=trj["xt"], sigh=trj["sig_h"], dt=trj["dt"])
        return like.sum() / weight, -np.hstack((np.zeros(self._model.force.size), jac.sum(axis=0) / weight))

    def e_step(self, weight, trj, coefficients, mu0, sig0):
        """
        In presence of hidden variables, reconstruct then using a Kalman Filter.
        Assume that the model is an OverdampedHidden model
        """
        self._model.coefficients = coefficients
        muh, Sigh = filtersmoother(
            trj["xt"][:, : self._model.dim_x],
            self._model.force(trj["x"][:, : self._model.dim_x], trj["bias"][:, : self._model.dim_x]) * trj["dt"],
            self._model.friction(trj["x"][:, : self._model.dim_x]) * trj["dt"],
            self._model.diffusion(trj["x"][:, : self._model.dim_x]) * trj["dt"],
            mu0,
            sig0,
        )

        trj["sig_h"] = Sigh
        trj["x"][:, self._model.dim_x :] = muh[:, self._model.dim_h :]
        trj["xt"][:, self._model.dim_x :] = muh[:, : self._model.dim_h]
        return muh[0, self._model.dim_h :] / weight, Sigh[0, self._model.dim_h :, self._model.dim_h :] / weight  # Return µ0 and sig0


class OzakiDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Ozaki approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity1D(self, x, xt, dt: float, bias=0.0, **kwargs):
        """
        The transition density obtained via Ozaki expansion
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """
        sig = self._model.diffusion(x, **kwargs).ravel()
        mu = self._model.meandispl(x, bias, **kwargs).ravel()
        mu_x = self._model.meandispl.grad_x(x, bias, **kwargs).ravel()
        temp = mu * (np.exp(mu_x * dt) - 1) / mu_x

        Mt = x.ravel() + temp
        Kt = (2 / dt) * np.log(1 + temp / x.ravel())
        Vt = np.sqrt(sig * (np.exp(Kt * dt) - 1) / Kt)

        return self._gaussian_likelihood_1D(xt, Mt, Vt)


class ShojiOzakiDensity(TransitionDensity):
    def __init__(self, model):
        """
        Class which represents the Shoji-Ozaki approximation transition density for a model
        :param model: the SDE model, referenced during calls to the transition density
        """
        super().__init__(model)

    def _logdensity1D(self, x, xt, dt: float, bias=0.0, **kwargs):
        """
        The transition density obtained via Shoji-Ozaki expansion
        :param x: float or array, the current value
        :param xt: float or array, the value to transition to  (must be same dimension as x)
        :param dt: float, the time step between x and xt
        :return: probability (same dimension as x and xt)
        """
        sig = np.sqrt(self._model.diffusion(x, **kwargs).ravel())
        mu = self._model.meandispl(x, bias, **kwargs).ravel()

        Mt = 0.5 * sig ** 2 * self._model.meandispl.hessian_x(x, bias, **kwargs).ravel()  # + self._model.meandispl_t(x)  #Time homogenous model
        Lt = self._model.meandispl.grad_x(x, bias, **kwargs).ravel()
        if (Lt == 0).any():  # TODO: need to fix this
            B = sig * np.sqrt(dt)
            A = x.ravel() + mu * dt + Mt * dt ** 2 / 2
        else:
            B = sig * np.sqrt((np.exp(2 * Lt * dt) - 1) / (2 * Lt))

            elt = np.exp(Lt * dt) - 1
            A = x.ravel() + mu / Lt * elt + Mt / (Lt ** 2) * (elt - Lt * dt)

        return self._gaussian_likelihood_1D(xt, A, B)


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
        sig_x = self._model.diffusion.grad_x(x, **kwargs).ravel()
        if isinstance(x, np.ndarray) and (sig_x == 0).any:
            return super()._logdensity1D(x=x, xt=xt, dt=dt, bias=bias, **kwargs)[0]

        sig = self._model.diffusion(x, **kwargs).ravel()
        mu = self._model.meandispl(x, bias, **kwargs).ravel()

        A = sig * sig_x * dt * 0.5
        B = -0.5 * sig / sig_x + x.ravel() + mu * dt - A
        z = (xt.ravel() - B) / A
        C = 1.0 / (sig_x ** 2 * dt)

        scz = np.sqrt(C * z)
        cpz = -0.5 * (C + z)
        ch = np.exp(scz + cpz) + np.exp(-scz + cpz)
        return -0.5 * np.log(z) + np.log(ch) - np.log(2 * np.abs(A) * np.sqrt(2 * np.pi))

    def __call__(self, weight, trj, coefficients):
        """
        Compute Likelihood of one trajectory
        """
        self._model.coefficients = coefficients
        return (-np.sum(np.maximum(self._min_prob, self._logdensity(x=trj["x"], xt=trj["xt"], dt=trj["dt"], bias=trj["bias"]))) / weight,)


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
        sig = self._model.diffusion(x, **kwargs).ravel()
        sig_x = self._model.diffusion.grad_x(x, **kwargs).ravel()
        sig_xx = self._model.diffusion.hessian_x(x, **kwargs).ravel()
        mu = self._model.meandispl(x, bias, **kwargs).ravel()
        mu_x = self._model.meandispl.grad_x(x, bias, **kwargs).ravel()
        mu_xx = self._model.meandispl.hessian_x(x, bias, **kwargs).ravel()
        x = x.ravel()
        d = dt ** 2 / 2
        E = x + mu * dt + (mu * mu_x + 0.5 * sig * mu_xx) * d

        V = x ** 2 + (2 * mu * x + sig) * dt + (2 * mu * (mu_x * x + mu + 0.5 * sig_x) + sig * (mu_xx * x + 2 * mu_x + 0.5 * sig_xx)) * d - E ** 2
        V = np.abs(V)
        return self._gaussian_likelihood_1D(xt, E, V)


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
        sig = self._model.diffusion(x, **kwargs).ravel()
        sig_x = self._model.diffusion.grad_x(x, **kwargs).ravel()
        sig_xx = self._model.diffusion.hessian_x(x, **kwargs).ravel()
        mu = self._model.meandispl(x, bias, **kwargs).ravel()
        mu_x = self._model.meandispl.grad_x(x, bias, **kwargs).ravel()
        mu_xx = self._model.meandispl.hessian_x(x, bias, **kwargs).ravel()

        d = dt ** 2 / 2
        E = x.ravel() + mu * dt + (mu * mu_x + 0.5 * sig * mu_xx) * d

        V = sig * dt + (mu * sig_x + 2 * mu_x * sig + sig * sig_xx) * d
        V = np.abs(V)
        return self._gaussian_likelihood_1D(xt, E, V)
