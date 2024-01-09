#!python3
# -*- coding: utf-8 -*-

"""
===========================
Functional basis set
===========================

In this example, we present a subset of implemented functional basis set.
"""

import numpy as np
import matplotlib.pyplot as plt

import folie.function_basis as bf

from scipy.interpolate import splrep

x_range = np.linspace(-2, 2, 30).reshape(-1, 1)

t, c, k = splrep(x_range, x_range ** 4 - 2 * x_range ** 2 + 0.5 * x_range)

basis_set = {
    "Linear": bf.Linear(),
    "Polynom": bf.Polynomial(3),
    "Hermite Polynom": bf.Polynomial(3, np.polynomial.Hermite),
    "Fourier": bf.Fourier(order=2, freq=1.0),
    "B Splines": bf.BSplines(6, k=3),
    "Splines Fct": bf.SplineFct(t, c, k),
}
for key, basis in basis_set.items():
    basis.fit(x_range)

fig_kernel, axs = plt.subplots(2, 3)
m = 0
for key, basis in basis_set.items():
    axs[m // 3][m % 3].set_title(key)
    axs[m // 3][m % 3].set_xlabel("$x$")
    axs[m // 3][m % 3].set_ylabel("$h_k(x)$")
    axs[m // 3][m % 3].grid()
    y = basis.transform(x_range)

    for n in range(y.shape[1]):
        axs[m // 3][m % 3].plot(x_range[:, 0], y[:, n])
    m += 1
plt.show()
