#!python3
# -*- coding: utf-8 -*-

"""
=======================================
Hidden Overdamped Langevin Estimation
=======================================

How to run a simple estimation
"""

import numpy as np
import matplotlib.pyplot as plt

import folie as fl

# Trouver comment on rentre les données
trj = np.loadtxt("datasets/example_2d.trj")
data = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
# for i in range(1, trj.shape[1]):
data.append(trj[:, 1:2])

fun_lin = fl.functions.Linear()
fun_cst = fl.functions.Constant()
model = fl.models.OverdampedHidden(fun_lin, fun_lin.copy(), fun_cst, dim=1, dim_h=2)
estimator = fl.EMEstimator(fl.EulerDensity(model), max_iter=3, verbose=2, verbose_interval=1)
model = estimator.fit_fetch(data)

# To find a correct parametrization of the space
bins = np.histogram_bin_edges(data[0]["x"], bins=15)
xfa = (bins[1:] + bins[:-1]) / 2.0


fig, axs = plt.subplots(1, 3)
# Force plot
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()
axs[0].plot(xfa, model.force(xfa.reshape(-1, 1)))


# Friction plot
axs[1].set_title("Friction")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$E_2(x)$")
axs[1].grid()
axs[1].plot(xfa, model.friction(xfa.reshape(-1, 1))[:, 0, :])

# Diffusion plot
axs[2].set_title("Diffusion")
axs[2].grid()
axs[2].plot(xfa, model.diffusion(xfa.reshape(-1, 1))[:, 0, 0])
axs[2].set_xlabel("$x$")
axs[2].set_ylabel("$D(x)$")
plt.show()
