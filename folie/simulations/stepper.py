from abc import ABC
from .._numpy import np


class Stepper(ABC):
    def __init__(self, model):
        """
        Class which represents the transition density for a model, and implements a __call__ method to evalute the
        transition density (bound to the model)

            Parameters
            ----------
            model: the SDE model, referenced during calls to the transition density
        """
        self._model = model

        if not hasattr(self, "run_step"):
            if self._model.dim <= 1:
                self.run_step = self.run_step_1D
            else:
                if not hasattr(self, "run_step_ND"):
                    raise ValueError("This stepper does not support multidimensionnal model.")
                self.run_step = self.run_step_ND

    @property
    def model(self):
        """Access to the underlying model"""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


class ExactStepper(Stepper):
    def run_step(self, x, dt, dW, bias=0.0):
        return self.model.exact_step(x, dt, dW)


class EulerStepper(Stepper):
    def run_step_1D(self, x, dt, dW, bias=0.0):
        sig_sq_dt = np.sqrt(self.model.diffusion(x) * dt)
        return (x.T + (self.model.meandispl(x, bias)) * dt + sig_sq_dt * dW).T  # Weird transpose for broadcasting
    
    def run_step_ND(self, x, dt, dW, bias=0.0):  # New function
        sig_sq_dt =np.sqrt(self.model.diffusion(x) * dt)  # Work only for diagonal diffusion that should a cholesky instead
        return x + self.model.meandispl(x, bias) * dt + np.einsum("ijk,ik->ij", sig_sq_dt, dW) 


class MilsteinStepper(Stepper):
    def run_step_1D(self, x, dt, dW, bias=0.0):
        sig_sq_dt = np.sqrt(self.model.diffusion(x) * dt)
        return (x.T + (self.model.meandispl(x, bias)) * dt + sig_sq_dt * dW + 0.25 * self.model.diffusion.grad_x(x)[..., 0] * (dW**2 - 1) * dt).T


class VECStepper(Stepper):
    def run_step(self, X, dt, dW, bias=0.0):
        x = X[:, : self.model.dim]
        v = X[:, self.model.dim :]
        # If not initialized initialize
        try:
            fx = self.f
            gamma = self.gamma
            diff = self.diff
        except AttributeError:
            fx = self.model.force(x, bias)
            gamma = self.model.friction(x)
            diff = self.model.diffusion(x)

        sc2 = 1 - 0.5 * gamma * dt + 0.125 * (gamma * dt) ** 2
        c1 = 0.5 * dt * (1 - 0.25 * gamma * dt)
        d1 = 0.5 * (1 - 0.25 * gamma * dt)
        d2 = -0.25 * gamma * dt / np.sqrt(3)
        sig_sq_dt = np.sqrt(diff * dt)
        dWx = sig_sq_dt * dW[:, : self.model.dim]
        dWv = sig_sq_dt * dW[:, self.model.dim :]
        v_mid = sc2 * v + c1 * fx + d1 * dWx + d2 * dWv
        x += dt * (v_mid + (0.5 / np.sqrt(3)) * dWv)
        # Update with new value of the field
        self.f = self.model.force(x, bias)
        self.gamma = self.model.friction(x)
        self.diff = self.model.diffusion(x)

        v = sc2 * v_mid + c1 * self.f + d1 * dWx + d2 * dWv
        return np.concatenate([x, v], axis=1)
