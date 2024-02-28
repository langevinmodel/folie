#!python3
# -*- coding: utf-8 -*-

"""
===========================
Functional set
===========================

In this example, we present a subset of implemented functions.
"""

import numpy as np
import matplotlib.pyplot as plt

import folie.functions as ff

from scipy.interpolate import splrep

x_range = np.linspace(-2, 2, 30).reshape(-1, 1)

t, c, k = splrep(x_range, x_range**4 - 2 * x_range**2 + 0.5 * x_range)

fun_set = {
    "Linear": ff.Linear(),
    "Polynom": ff.Polynomial(3),
    "Hermite Polynom": ff.Polynomial(3, np.polynomial.Hermite),
    "Fourier": ff.Fourier(order=2, freq=1.0),
    "B Splines": ff.BSplinesFunction(knots=6, k=3),
}
for key, fun in fun_set.items():
    fun.fit(x_range)

fig_kernel, axs = plt.subplots(2, 3)
m = 0
for key, fun in fun_set.items():
    axs[m // 3][m % 3].set_title(key)
    axs[m // 3][m % 3].set_xlabel("$x$")
    axs[m // 3][m % 3].set_ylabel("$h_k(x)$")
    axs[m // 3][m % 3].grid()
    y = fun.grad_coeffs(x_range)

    for n in range(y.shape[1]):
        axs[m // 3][m % 3].plot(x_range[:, 0], y[:, n])
    m += 1
plt.show()
