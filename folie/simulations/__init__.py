"""
Module that contains some utils some basic simulations of the models. Mostly for short validations runs.
For more efficient and longuer simulations, please turn to LangevinIntegrators.jl or StochasticDiffEq.jl

ABMD simulation is adapted from pyoptLE
"""

from .._numpy import np
from ..data import Trajectories, Trajectory
from .stepper import ExactStepper, EulerStepper, MilsteinStepper, VECStepper


class Simulator:
    def __init__(self, stepper, dt, seed=None, keep_dim=None):
        self.dt = dt
        self.stepper = stepper
        self.keep_dim = keep_dim
        dim = self.stepper.model.dim

        if keep_dim is None:
            self.keep_dim = dim
        else:
            self.keep_dim = self.keep_dim % dim

    def run(self, nsteps, x0, save_every=1, **kwargs):
        dim = self.stepper.model.dim
        x = np.asarray(x0).reshape(-1, dim)
        ntrajs = x.shape[0]

        x_val = np.empty((ntrajs, nsteps // save_every, dim))
        for n in range(nsteps):
            dW = np.random.normal(loc=0.0, scale=1.0, size=(ntrajs, dim))
            x = self.stepper.run_step(x, self.dt, dW)
            if n % save_every == 0:
                x_val[:, n // save_every, :] = x
        data = Trajectories(dt=self.dt * save_every)
        for i in range(ntrajs):
            data.append(x_val[i, :, : self.keep_dim])
        return data


class UnderdampedSimulator(Simulator):
    def run(self, nsteps, x0, save_every=1, **kwargs):
        dim = 2 * self.stepper.model.dim
        x = np.asarray(x0).reshape(-1, dim)
        ntrajs = x.shape[0]

        x_val = np.empty((ntrajs, nsteps // save_every, dim))
        for n in range(nsteps):
            dW = np.random.normal(loc=0.0, scale=1.0, size=(ntrajs, dim))
            x = self.stepper.run_step(x, self.dt, dW)
            if n % save_every == 0:
                x_val[:, n // save_every, :] = x
        data = Trajectories(dt=self.dt * save_every)
        for i in range(ntrajs):
            data.append(Trajectory(self.dt, x_val[i, :, : self.keep_dim], v=x_val[i, :, dim // 2 : dim // 2 + self.keep_dim]))
        return data


class BiasedSimulator(Simulator):
    def __init__(self, stepper, dt, k=1, **kwargs):
        super().__init__(stepper, dt, **kwargs)
        self.stepper.model.add_bias()

    def run(self, nsteps, x0, save_every=1, **kwargs):
        dim = self.stepper.model.dim
        x = np.asarray(x0).reshape(-1, dim)
        ntrajs = x.shape[0]

        x_val = np.empty((ntrajs, nsteps // save_every, dim))
        bias_t = np.empty((ntrajs, nsteps // save_every, dim))
        dW = np.random.normal(loc=0.0, scale=1.0, size=(ntrajs, nsteps))
        for n in range(nsteps):
            bias = self._bias(x)
            x = self.stepper.run_step(x, self.dt, dW[:, n], bias=bias)
            if n % save_every == 0:
                x_val[:, n // save_every, :] = x
                bias_t[:, n // save_every, :] = bias
        data = Trajectories(dt=self.dt * save_every)
        for i in range(ntrajs):
            data.append(Trajectory(self.dt, x_val[i, :, : self.keep_dim], bias=bias_t[i, :, : self.keep_dim]))
        return data

    def _bias(self, xt):
        return 0.0


class ABMD_Simulator(BiasedSimulator):
    def __init__(self, stepper, dt, k=1, xstop=np.infty, **kwargs):
        super().__init__(stepper, dt, **kwargs)
        self.xmax = None
        self.k = k
        self.xstop = xstop
        self.xmax_hist = []

    def _bias(self, xt):
        if self.xmax is None:
            self.xmax = np.copy(xt)
        else:
            np.maximum(self.xmax, xt, out=self.xmax)
        np.minimum(self.xmax, self.xstop, out=self.xmax)
        self.xmax_hist.append(np.copy(self.xmax))
        return self.k * (self.xmax - xt)
    

class ABMD_Colvar_Simulator(BiasedSimulator):
    def __init__(self, stepper, dt, colvar, k=1, xstop=np.infty, **kwargs):
        super().__init__(stepper, dt, **kwargs)
        self.xmax = None
        self.k = k
        self.xstop = xstop
        self.xmax_hist = []
        self.colvar = colvar

    def _bias(self, xt):
        # q_x: gradient
        q, q_x = self.colvar(xt)
        if self.qmax is None:
            self.qmax = np.copy(q)
        else:
            np.maximum(self.xmax, xt, out=self.xmax)
        np.minimum(self.xmax, self.xstop, out=self.xmax)
        self.xmax_hist.append(np.copy(self.xmax))
        return (self.k * (self.xmax - xt)) * q_x
    
