import pytest
import numpy as np


def test_inversion(benchmark):
    a = np.linspace(1.0, 2.0, 5000)
    sig = np.linspace(2.0, 3.0, 5000)
    res = benchmark(lambda x, b: a ** 2 / x, sig, a)
    assert res.shape == (5000,)


def test_linalg_inversion(benchmark):
    a = np.linspace(1.0, 2.0, 5000).reshape(-1, 1)
    sig = np.linspace(2.0, 3.0, 5000).reshape(-1, 1, 1)

    def ll(E, V):
        invV = np.linalg.inv(V)
        return -0.5 * np.einsum("ti,tij,tj-> t", E, invV, E)

    res = benchmark(ll, a, sig)
    assert res.shape == (5000,)


def test_solve(benchmark):
    a = np.linspace(1.0, 2.0, 5000).reshape(-1, 1)
    sig = np.linspace(2.0, 3.0, 5000).reshape(-1, 1, 1)

    def ll(E, V):
        invV = np.linalg.solve(V, E)
        return -0.5 * np.einsum("ti,tj-> t", E, invV)

    res = benchmark(ll, a, sig)
    assert res.shape == (5000,)
