# A set of analytical functions for examples
from .base import Function
from .._numpy import np
from ..domains import Domain


class PotentialFunction(Function):
    def __init__(self):
        if self.dim == 1:
            output_shape = ()
        else:
            output_shape = (self.dim,)
        domain = Domain.Rd(self.dim)
        super().__init__(domain, output_shape)

    def resize(self, new_shape):
        if new_shape != (self.dim,) or (self.dim == 1 and (new_shape != ())):
            print("Analytical functions cannot be resized, check what your are doing")
        return self

    def transform(self, x, *args, **kwargs):
        return self.force(x)


class ConstantForce(PotentialFunction):
    """
    Constant force potential
    """

    def __init__(self, a=1.0, dim=1):
        self.a = a * np.ones((dim,))
        self.dim = dim
        super().__init__()

    @property
    def coefficients(self):
        return np.array([self.a])

    @coefficients.setter
    def coefficients(self, val):
        self.a = val[0]

    def potential(self, x):
        return np.dot(x, self.a)

    def force(self, X):
        return -self.a * np.ones_like(X)


class Quadratic(PotentialFunction):
    """
    Quadratic potential
    """

    def __init__(self, a=0.5):
        self.dim = 1
        self.a = a
        super().__init__()

    @property
    def coefficients(self):
        return np.array([self.a])

    @coefficients.setter
    def coefficients(self, val):
        self.a = val[0]

    def potential(self, x):
        return 0.5 * self.a * np.sum(x**2, axis=1)

    def force(self, X):
        return -1 * self.a * X


class Quartic(PotentialFunction):
    """
    Quartic potential with minima at x0 and x1
    """

    def __init__(self, a=25, x0=0.0, x1=1.0):
        self.a = a
        self.x0 = x0
        self.x1 = x1
        self.dim = 1
        super().__init__()

    @property
    def coefficients(self):
        return np.array([self.a, self.x0, self.x1])

    @coefficients.setter
    def coefficients(self, val):
        self.a, self.x0, self.x1 = val

    def potential(self, x):
        return np.sum(self.a * ((x - self.x0) * (x - self.x1)) ** 2, axis=1)

    def force(self, x):
        return -2 * self.a * ((x - self.x0) * (x - 1.0)) * (2 * x - self.x0 - self.x1)


class Cosine(PotentialFunction):
    """
    Cosine potential
    """

    def __init__(self, a=1.0):
        self.a = a
        self.dim = 1
        super().__init__()

    @property
    def coefficients(self):
        return np.array([self.a])

    @coefficients.setter
    def coefficients(self, val):
        self.a = val[0]

    def potential(self, x):
        return self.a * (1 - np.cos(x[:, 0]))

    def force(self, X):
        return -self.a * np.sin(X)


class Quartic2D(PotentialFunction):
    """
    Simple 2D potential that is quadratic in one axis and quartic in the other one
    """

    def __init__(self, a=2, b=1.0):
        self.a = a
        self.b = b
        self.dim = 2
        super().__init__()

    @property
    def coefficients(self):
        return np.array([self.a, self.b])

    @coefficients.setter
    def coefficients(self, val):
        self.a, self.b = val

    def potential(self, X):
        return self.a * (X[:, 0] ** 2 - 1.0) ** 2 + 0.5 * self.b * X[:, 1] ** 2

    def force(self, X):
        F = np.zeros(X.shape)
        F[:, 0] = -4 * self.a * X[:, 0] * (X[:, 0] ** 2 - 1.0)
        F[:, 1] = -self.b * X[:, 1]
        return F


class ThreeWell(PotentialFunction):
    """
    A three well potential
    """

    def __init__(self, Ac=[3.0, -3.0, -5.0, -5.0], a=[-1, -1, -1, -1], b=[0, 0, 0, 0], c=[-1, -1, -1, -1], x0=[0, 0, 1, -1], y0=[1.0 / 3, 5.0 / 3, 0, 0]):
        self.Ac = 2 * np.array(Ac)
        self.a = a
        self.b = b
        self.c = c
        self.x0 = x0
        self.y0 = y0

        self.dim = 2
        super().__init__()

    @property
    def coefficients(self):
        return np.array([*self.Ac, *self.a, *self.b, *self.c, *self.x0, *self.y0])

    def potential(self, X):
        """
        Compute three wells potential
        """
        x = X[:, 0]
        y = X[:, 1]
        pot = 0.0
        for n in range(4):
            pot += self.Ac[n] * np.exp(self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (x - self.x0[n]) * (y - self.y0[n]) + self.c[n] * (y - self.y0[n]) ** 2)
        pot += 2 * x**4 / 10 + 2 * (y - 1.0 / 3) ** 4 / 10
        return pot

    def force(self, X):
        """
        Compute potential derivative
        """
        x = X[:, 0]
        y = X[:, 1]

        dU = np.zeros(X.shape)
        for n in range(4):
            dU[:, 0] += self.Ac[n] * (2 * self.a[n] * (x - self.x0[n]) + self.b[n] * (y - self.y0[n])) * np.exp(self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (x - self.x0[n]) * (y - self.y0[n]) + self.c[n] * (y - self.y0[n]) ** 2)
            dU[:, 1] += self.Ac[n] * (self.b[n] * (x - self.x0[n]) + 2 * self.c[n] * (y - self.y0[n])) * np.exp(self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (x - self.x0[n]) * (y - self.y0[n]) + self.c[n] * (y - self.y0[n]) ** 2)
        dU[:, 0] += 8 * x**3 / 10
        dU[:, 1] += 8 * (y - 1.0 / 3) ** 3 / 10
        return -dU


def EntropicSwitch():
    return ThreeWell()


class MullerBrown(PotentialFunction):
    """
    Müller Brown model
    """

    def __init__(self, Ac=[-20.0, -10.0, -17.0, 1.5], a=[-1, -1, -6.5, 0.7], b=[0, 0, 11, 0.6], c=[-10, -10, -6.5, 0.7], x0=[1, 0, -0.5, -1], y0=[0, 0.5, 1.5, 1]):
        self.Ac = 2 * np.array(Ac)
        self.a = a
        self.b = b
        self.c = c
        self.x0 = x0
        self.y0 = y0

        self.dim = 2
        super().__init__()

    @property
    def coefficients(self):
        return np.array([*self.Ac, *self.a, *self.b, *self.c, *self.x0, *self.y0])

    def potential(self, X):
        """
        Compute muller brown potential
        """
        pot = 0.0
        for n in range(4):
            pot += self.Ac[n] * np.exp(self.a[n] * (X[:, 0] - self.x0[n]) ** 2 + self.b[n] * (X[:, 0] - self.x0[n]) * (X[:, 1] - self.y0[n]) + self.c[n] * (X[:, 1] - self.y0[n]) ** 2)
        return pot

    def force(self, X):
        """
        Compute potential derivative
        """
        x = X[:, 0]
        y = X[:, 1]
        dU = np.zeros(X.shape)
        for n in range(4):
            dU[:, 0] += self.Ac[n] * (2 * self.a[n] * (x - self.x0[n]) + self.b[n] * (y - self.y0[n])) * np.exp(self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (x - self.x0[n]) * (y - self.y0[n]) + self.c[n] * (y - self.y0[n]) ** 2)
            dU[:, 1] += self.Ac[n] * (self.b[n] * (x - self.x0[n]) + 2 * self.c[n] * (y - self.y0[n])) * np.exp(self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (x - self.x0[n]) * (y - self.y0[n]) + self.c[n] * (y - self.y0[n]) ** 2)
        return -dU


class RuggedMullerBrown(PotentialFunction):
    """
    Müller Brown model
    """

    def __init__(self, Ac=[-200.0, -100.0, -170.0, 15], a=[-1, -1, -6.5, 0.7], b=[0, 0, 11, 0.6], c=[-10, -10, -6.5, 0.7], x0=[1, 0, -0.5, -1], y0=[0, 0.5, 1.5, 1], gamma=9, k=5):
        self.Ac = 2 * np.array(Ac)
        self.a = a
        self.b = b
        self.c = c
        self.x0 = x0
        self.y0 = y0

        self.gamma = gamma
        self.k = k

        self.dim = 2
        super().__init__()

    @property
    def coefficients(self):
        return np.array([*self.Ac, *self.a, *self.b, *self.c, *self.x0, *self.y0, self.gamma, self.k])

    def potential(self, X):
        """
        Compute muller brown potential
        """
        pot = 0.0
        for n in range(4):
            pot += self.Ac[n] * np.exp(self.a[n] * (X[:, 0] - self.x0[n]) ** 2 + self.b[n] * (X[:, 0] - self.x0[n]) * (X[:, 1] - self.y0[n]) + self.c[n] * (X[:, 1] - self.y0[n]) ** 2)
        pot += self.gamma * np.sin(2 * self.k * np.pi * X[:, 0]) * np.sin(2 * self.k * np.pi * X[:, 1])
        return pot

    def force(self, X):
        """
        Compute potential derivative
        """
        x = X[:, 0]
        y = X[:, 1]

        dU = np.zeros(X.shape)
        for n in range(4):
            dU[:, 0] += self.Ac[n] * (2 * self.a[n] * (x - self.x0[n]) + self.b[n] * (y - self.y0[n])) * np.exp(self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (x - self.x0[n]) * (y - self.y0[n]) + self.c[n] * (y - self.y0[n]) ** 2)
            dU[:, 1] += self.Ac[n] * (self.b[n] * (x - self.x0[n]) + 2 * self.c[n] * (y - self.y0[n])) * np.exp(self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (x - self.x0[n]) * (y - self.y0[n]) + self.c[n] * (y - self.y0[n]) ** 2)
        dU[:, 0] += 2 * self.k * np.pi * self.gamma * np.cos(2 * self.k * np.pi * x) * np.sin(2 * self.k * np.pi * y)
        dU[:, 1] += 2 * self.k * np.pi * self.gamma * np.sin(2 * self.k * np.pi * x) * np.cos(2 * self.k * np.pi * y)
        return -dU


class LogExpPot(PotentialFunction):
    """
    Simple multiwell potential
    """

    three_well = {"a": [0.1, 0.15, 0.04], "x0": [5.0, 25.0, 60.0], "y0": [5.0, 5.0, 10.0]}
    four_well = {"a": [0.1, 0.2, 0.2, 0.05], "x0": [1.0, 20.0, 5.0, 30], "y0": [5.0, 7.0, 30.0, 35]}

    def __init__(self, a=[0.1, 0.15, 0.04], b=None, x0=[5.0, 25.0, 60.0], y0=[5.0, 5.0, 10.0]):

        self.a = a
        if b is None:
            self.b = a
        else:
            self.b = b
        self.x0 = x0
        self.y0 = y0
        self.n_well = len(self.a)

        self.dim = 2
        super().__init__()

    @property
    def coefficients(self):
        return np.array([*self.a, *self.b, *self.x0, *self.y0])

    def potential(self, X):
        """
        Compute muller brown potential
        """
        pot = 0.0
        for n in range(self.n_well):
            pot += np.exp(-0.5 * (self.a[n] * (X[:, 0] - self.x0[n]) ** 2 + self.b[n] * (X[:, 1] - self.y0[n]) ** 2))
        return -np.log(pot)

    def force(self, X):
        """
        Compute potential derivative
        """
        x = X[:, 0]
        y = X[:, 1]
        dU = np.zeros(X.shape)
        norm = 0.0
        for n in range(self.n_well):
            dU[:, 0] += self.a[n] * (x - self.x0[n]) * np.exp(-0.5 * (self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (y - self.y0[n]) ** 2))
            dU[:, 1] += self.b[n] * (y - self.y0[n]) * np.exp(-0.5 * (self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (y - self.y0[n]) ** 2))
            norm += np.exp(-0.5 * (self.a[n] * (x - self.x0[n]) ** 2 + self.b[n] * (y - self.y0[n]) ** 2))
        return -1 * np.divide(dU, norm[:, None], out=np.zeros_like(dU), where=norm[:, None] != 0)


class ValleyRidgePotential(PotentialFunction):
    """
    Simple potential with a valley ridge inflexion point
    """

    def __init__(self, xw=1.25, yw=1.0, Hw=-1.0, xs=1.0, xi=0.5, V=0.5):

        self.xs = xs
        self.xi = xi
        self.V = V

        b1 = Hw - self.V * (xw / self.xs) ** 2 * ((xw / self.xs) ** 2 - 2)
        b2 = -4 * self.V * xw * (self.xs**2 - xw**2) / self.xs**4

        A = [
            [yw**2 * (self.xi - xw), yw**4, yw**4 * xw],
            [yw**2, 0.0, -(yw**4)],
            [2 * yw * (xw - self.xi), -4 * yw**3, -4 * yw**3 * xw],
        ]

        par = np.linalg.solve(A, np.array([b1, b2, 0.0]))

        self.A = par[0]
        self.B = par[1]
        self.C = par[2]

        self.dim = 2
        super().__init__()

    @property
    def coefficients(self):
        return np.array([self.V, self.A, self.B, self.C, self.xs, self.xi])

    def potential(self, X):
        """
        Compute muller brown potential
        """

        return self.V * (X[:, 0] / self.xs) ** 2 * ((X[:, 0] / self.xs) ** 2 - 2) + self.A * X[:, 1] ** 2 * (self.xi - X[:, 0]) + X[:, 1] ** 4 * (self.B + self.C * X[:, 0])

    def force(self, X):
        """
        Compute potential derivative
        """
        x = X[:, 0]
        y = X[:, 1]
        F = np.zeros(X.shape)
        F[:, 0] = 4 * self.V * x * (self.xs**2 - x**2) / self.xs**4 + self.A * y**2 - self.C * y**4
        F[:, 1] = 2 * self.A * y * (x - self.xi) - 4 * y**3 * (self.B + self.C * x)
        return F


class SimpleValleyRidgePotential(PotentialFunction):
    """
    Simple potential with a valley ridge inflexion point
    """

    def __init__(self, a=0.5):
        self.a = a
        self.dim = 2
        super().__init__()

    @property
    def coefficients(self):
        return np.array([self.a])

    def potential(self, X):
        """
        Compute muller brown potential
        """
        pot = 8 * X[:, 0] ** 3 / 3.0 - 4 * X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2 + X[:, 0] * X[:, 1] ** 2 * (X[:, 1] ** 2 - 2)
        return self.a * pot

    def force(self, X):
        """
        Compute potential derivative
        """
        x = X[:, 0]
        y = X[:, 1]
        F = np.zeros(X.shape)
        F[:, 0] = 8 * x * (1 - x) + y**2 * (2 - y**2)
        F[:, 1] = y * (4 * x * (1 - y**2) - 1)
        return self.a * F
