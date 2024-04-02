#!python3
# -*- coding: utf-8 -*-

"""
================================
Overdamped Langevin Estimation
================================

How to run a simple estimation
"""

import numpy as np
import matplotlib.pyplot as plt

import folie as fl

fig, axs = plt.subplots(2, 2)

# Trouver comment on rentre les données
trj = np.loadtxt("datasets/example_underdamped2.trj")
data = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
data.append(trj[:, 1:2])

axs[0, 0].set_title("Phase Space")
axs[0, 0].set_xlabel("$x$")
axs[0, 0].set_ylabel("$v$")
axs[0, 0].grid()
axs[0, 0].plot(trj[:, 1:2], trj[:, 2:3])


fun = fl.functions.Polynomial(deg=3)
fun_cst = fl.functions.Constant()
model = fl.models.Underdamped(fun, fun_cst, fun_cst)
estimator = fl.UnderdampedKramersMoyalEstimator(model)
model = estimator.fit_fetch(data)
print(model.coefficients)

# To find a correct parametrization of the space
xfa = np.linspace(data.stats.min, data.stats.max, 150).ravel()

# Force plot
axs[0, 1].set_title("Force")
axs[0, 1].set_xlabel("$x$")
axs[0, 1].set_ylabel("$F(x)$")
axs[0, 1].grid()
axs[0, 1].plot(xfa, model.force(xfa.reshape(-1, 1)))
axs[0, 1].plot(xfa, -2 * xfa * (xfa**2 - 1), label="True")


# Friction plot
axs[1, 0].set_title("Friction")
axs[1, 0].set_xlabel("$x$")
axs[1, 0].set_ylabel("$\\gamma(x)$")
axs[1, 0].grid()
axs[1, 0].plot(xfa, model.friction(xfa.reshape(-1, 1)))
axs[1, 0].plot(xfa, 0.1 * np.ones_like(xfa))


# Diffusion plot
axs[1, 1].set_title("Diffusion")
axs[1, 1].grid()
axs[1, 1].plot(xfa, model.diffusion(xfa.reshape(-1, 1)))
axs[1, 1].plot(xfa, 0.1 / 3.0 * np.ones_like(xfa))
axs[1, 1].set_xlabel("$x$")
axs[1, 1].set_ylabel("$D(x)$")
plt.show()
