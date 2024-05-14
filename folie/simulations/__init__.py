"""
Module that contains some utils some basic simulations of the models. Mostly for short validations runs.
For more efficient and longuer simulations, please turn to LangevinIntegrators.jl or StochasticDiffEq.jl

ABMD simulation is adapted from pyoptLE
"""

from .._numpy import np
from ..data import Trajectories, Trajectory
from .stepper import ExactStepper, EulerStepper, MilsteinStepper, VECStepper
import sympy as sym
# np.random.seed(3)

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
        for n in range(nsteps):
            bias = self._bias(x)
            dW = np.random.normal(loc=0.0, scale=1.0, size=(ntrajs, dim)) # mod to mimic the Simulator class
            x = self.stepper.run_step(x, self.dt, dW, bias=bias)    
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
    

class ABMD_2D_to_1DColvar_Simulator_Generic(BiasedSimulator): #takes only function colvar as input, then symbolically computes the gradient
    def __init__(self, stepper, dt, colvar, k=1, qstop=np.infty, **kwargs):
        super().__init__(stepper, dt, **kwargs)
        self.qmax = None
        self.k = k
        self.qstop = qstop
        self.qmax_hist = []
        self.colvar = colvar
    def _bias(self, xt):
        xa = sym.Symbol('xb')     #define sympy.Symbols to be used in symbolic manipulation
        ya = sym.Symbol('yb')     #define sympy.Symbols to be used in symbolic manipulation
        grad_q = np.empty_like(xt) # initalises an array who will host grad_xy(q) for each trajectory in each entry
        q=np.empty_like(xt[:,0])
        for i in range(len(xt)):
            q[i] = self.colvar(xa,ya).evalf(subs={xa: xt[i,0], ya:xt[i,1]})
            q_x=sym.diff(self.colvar(xa,ya),xa).evalf(subs={xa:xt[i,0],ya:xt[i,1]})     #calculates dq/dx and evaluates it in the current (x,y) position CAREFUL: It returns sympy.core.numbers.Float
            q_y=sym.diff(self.colvar(xa,ya),ya).evalf(subs={xa:xt[i,0],ya:xt[i,1]})     #calculates dq/dy and evaluates it in the current (x,y) position CAREFUL: It returns sympy.core.numbers.Float
            grad_q[i]=(float(q_x),float(q_y))                                           # append everything in the gradient array (need to cast sympy.core.numbers.Float type object to float)

        if self.qmax is None:
            self.qmax = np.copy(q)
        else:
            np.maximum(self.qmax, q, out=self.qmax)
        np.minimum(self.qmax, self.qstop, out=self.qmax)
        self.qmax_hist.append(np.copy(self.qmax))        
        return (self.k * (self.qmax - q)).reshape(len(q),1) *grad_q
    
class ABMD_2D_to_1DColvar_Simulator_Fast(BiasedSimulator): # needs also gradient of colvar as input, works only with constant gradient
    def __init__(self, stepper, dt, colvar, grad_colvar, k=1, qstop=np.infty, **kwargs):
        super().__init__(stepper, dt, **kwargs)
        self.qmax = None
        self.k = k
        self.qstop = qstop
        self.qmax_hist = []
        self.colvar = colvar
        self.grad_colvar = grad_colvar

    def _bias(self, xt):
        q = self.colvar(xt[:,0],xt[:,1])
        if self.qmax is None:
            self.qmax = np.copy(q)
        else:
            np.maximum(self.qmax, q, out=self.qmax)
        np.minimum(self.qmax, self.qstop, out=self.qmax)
        self.qmax_hist.append(np.copy(self.qmax))

        return (self.k * (self.qmax - q)).reshape(len(q),1) * self.grad_colvar