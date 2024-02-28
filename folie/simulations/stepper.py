from abc import ABC
import numpy as np


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
        return self._model.exact_step(x, dt, dW)


class EulerStepper(Stepper):
    def run_step_1D(self, x, dt, dW, bias=0.0):
        sig_sq_dt = np.sqrt(self._model.diffusion(x) * dt)
        return (x.T + (self._model.meandispl(x, bias)) * dt + sig_sq_dt * dW).T  # Weird transpose for broadcasting


class MilsteinStepper(Stepper):
    def run_step_1D(self, x, dt, dW, bias=0.0):
        sig_sq_dt = np.sqrt(self._model.diffusion(x) * dt)
        return (x.T + (self._model.meandispl(x, bias)) * dt + sig_sq_dt * dW + 0.25 * self._model.diffusion.grad_x(x)[..., 0] * (dW ** 2 - 1) * dt).T
