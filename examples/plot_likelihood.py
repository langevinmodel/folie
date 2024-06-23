#!python3
# -*- coding: utf-8 -*-

"""
===========================
Likelihood functions
===========================

A set of likelihood functions used for estimation
"""

import numpy as np
import matplotlib.pyplot as plt

import folie as fl

# Trouver comment on rentre les données
trj = np.loadtxt("datasets/example_2d.trj")
data = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
data.append(trj[:, 1:2])

model = fl.models.BrownianMotion()

fig, axs = plt.subplots(1, 2)
axs[0].set_title("Drift")
axs[0].set_xlabel("$f$")
axs[0].set_ylabel("$L(f,1.0)$")
axs[0].grid()


axs[1].set_title("Diffusion")
axs[1].grid()
axs[1].set_xlabel("$\\sigma$")
axs[1].set_ylabel("$L(1.0,\\sigma)$")


drift_range = np.linspace(-1, 1, 25)
diff_range = np.linspace(0.5, 2, 25)


for name, transitioncls in zip(
    ["Euler", "Elerian", "Kessler", "Drozdov"],
    [
        fl.EulerDensity,
        fl.ElerianDensity,
        fl.KesslerDensity,
        fl.DrozdovDensity,
    ],
):
    likelihood = transitioncls(model)
    likelihood.preprocess_traj(data[0])
    likelihood_vals_drift = np.zeros_like(drift_range)
    for n, f in enumerate(drift_range):
        likelihood_vals_drift[n] = likelihood(1.0, data[0], np.array([f, 1.0]))[0]
    axs[0].plot(drift_range, likelihood_vals_drift, label=name)
    likelihood_vals_diff = np.zeros_like(diff_range)
    for n, d in enumerate(diff_range):
        likelihood_vals_diff[n] = likelihood(1.0, data[0], np.array([1.0, d]))[0]

    axs[1].plot(diff_range, likelihood_vals_diff, label=name)

axs[0].legend()
axs[1].legend()
plt.show()
