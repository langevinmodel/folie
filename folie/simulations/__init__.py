"""
Module that contains some utils some basic simulations of the models. Mostly for short validations runs.
For more efficient and longuer simulations, please turn to LangevinIntegrators.jl or StochasticDiffEq.jl
"""

import numpy as np
from ..data import Trajectories


class Simulator:
    def __init__(self, transition, dt, seed=None, keep_dim=None):
        self.dt = dt
        self.transition = transition
        self.keep_dim = keep_dim

    def run(self, nsteps, x0, ntrajs=1, save_every=1, **kwargs):
        x0 = np.asarray(x0)
        if x0.shape[0] != ntrajs:
            raise ValueError("You must provide as much initial condtion as the wanted number of trajectories.")
        if x0.ndim == 1:
            dim = 1
        else:
            dim = x0.shape[1]
        if self.keep_dim is None:
            keep_dim = dim
        else:
            keep_dim = self.keep_dim % dim
        x = x0.reshape(-1)
        x_val = np.empty((ntrajs, nsteps // save_every, dim))
        dW = np.random.normal(loc=0.0, scale=1.0, size=(ntrajs, nsteps))
        for n in range(nsteps):
            x = self.transition.run_step(x, self.dt, dW[:, n])
            if n % save_every == 0:
                x_val[:, n // save_every, 0] = x
        data = Trajectories(dt=self.dt * save_every)
        for i in range(ntrajs):
            data.append(x_val[i, :, :keep_dim])
        return data


class BiasedSimulator(Simulator):

    def run(self, nsteps, x0, ntrajs=1, save_every=1, **kwargs):
        if x0.shape[0] != ntrajs:
            raise ValueError("You must provide as much initial condtion as the wanted number of trajectories.")
        if x0.ndim == 1:
            dim = 1
        else:
            dim = x0.shape[1]
        if self.keep_dim is None:
            keep_dim = dim
        else:
            keep_dim = self.keep_dim % dim
        x = x0.reshape(-1)
        x_val = np.empty((ntrajs, nsteps // save_every, dim))
        bias_t = np.empty((ntrajs, nsteps // save_every, dim))
        dW = np.random.normal(loc=0.0, scale=1.0, size=(ntrajs, nsteps))
        for n in range(nsteps):
            bias = self._bias(x)
            x = self.transition.run_step(x, self.dt, dW[:, n], bias=bias)
            if n % save_every == 0:
                x_val[:, n // save_every, 0] = x
                bias_t[:, n // save_every, 0] = bias
        data = Trajectories(dt=self.dt * save_every)
        for i in range(ntrajs):
            data.append(x_val[i, :, :keep_dim])  # Add also bias
        return data


class ABMD_Simulator(BiasedSimulator):
    def __init__(self, transition, dt, k=1, xstop=np.infty, **kwargs):
        super().__init__(transition, dt, **kwargs)
        self.xmax = None
        self.k = k
        self.xstop = xstop

    def _bias(self, xt):

        if self.xmax is None:
            self.xmax = np.copy(xt)
        else:
            np.maximum(self.xmax, xt, out=self.xmax)
        np.minimum(self.xmax, self.xstop, out=self.xmax)
        return self.k * (self.xmax - xt)
