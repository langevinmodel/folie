"""
Module that contains some utils some basic simulations of the model. Mostly for validations runs
For more efficient and longuer simulations, please turn to LangevinIntegrators.jl or StochasticDiffEq.jl
"""
import numpy as np
from ..data import Trajectories


class Simulator:
    def __init__(self, transition, dt, seed=None):
        self.dt = dt
        self.transition = transition

    def run(self, nsteps, x0, ntrajs=1, save_every=1, **kwargs):
        if x0.shape[0] != ntrajs:
            raise ValueError("You must provide as much initial condtion as the wanted number of trajectories.")
        if x0.ndim == 1:
            dim = 1
        else:
            dim = x0.shape[1]
        x = x0.reshape(-1)
        x_val = np.empty((ntrajs, nsteps // save_every, dim))
        dW = np.random.normal(loc=0.0, scale=1.0, size=(ntrajs, nsteps))
        for n in range(nsteps):
            x = self.transition.run_step(x, self.dt, dW[:, n])
            if n % save_every == 0:
                x_val[:, n // save_every, 0] = x
        data = Trajectories(dt=self.dt * save_every)
        for i in range(ntrajs):
            data.append(x_val[i, :, :])
        return data
