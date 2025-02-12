"""
================================
ABMD biased dynamics
================================

Estimation of an overdamped Langevin in presence of biased dynamics.
"""

import numpy as np
import folie as fl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# First let's generate some biased trajectories
potential = fl.functions.MultiWell1D(V=4.0)

x_range = np.linspace(0, 30, 150)
# plt.plot(x_range, potential.potential_plot(x_range.reshape(-1, 1)))
# plt.show()

diff_function = fl.functions.Polynomial(deg=0, coefficients=np.array(1.0))
model_simu = fl.Overdamped(potential, diffusion=diff_function)
simulator = fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu), 1e-3)
data = simulator.run(250000, 30 * np.random.rand(25), 1)

# Plot the resulting trajectories
# sphinx_gallery_thumbnail_number = 1
# fig, axs = plt.subplots(1, 1)
# for n, trj in enumerate(data):
#     axs[0].plot(trj["x"])

# axs[0].set_title("Trajectories")
# axs[0].set_xlabel("step")


fig, axs = plt.subplots(1, 3)
axs[0].set_title("Drift")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$F(x)$")
axs[0].grid()

axs[1].set_title("Diffusion")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$D(x)$")
axs[1].grid()

axs[2].set_title("Free energy")
axs[2].set_xlabel("$x$")
axs[2].set_ylabel("$V(x)$")
axs[2].grid()


xfa = np.linspace(0.0, 30.0, 75)
axs[0].plot(xfa, model_simu.drift(xfa.reshape(-1, 1)), label="Exact")
axs[1].plot(xfa, model_simu.diffusion(xfa.reshape(-1, 1)), label="Exact")


axs[2].plot(x_range, potential.potential_plot(x_range.reshape(-1, 1)), label="Exact")


name = "KramersMoyal"
model = fl.OverdampedSplines1D(fl.domains.MeshedDomain1D.create_from_range(np.linspace(-5, 35, 15)))
estimator = fl.KramersMoyalEstimator(model)
res = estimator.fit_fetch(data)
print(name, res.coefficients)
axs[0].plot(xfa, res.drift(xfa.reshape(-1, 1)), "--", label=name)
axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), "--", label=name)

# Compare Free energy
axs[2].plot(xfa, fl.analysis.free_energy_profile_1d(res, xfa), "--", label=name)


for name, marker, transitioncls in zip(
    ["Euler", "Elerian", "Kessler", "Drozdov"],
    ["+", "1", "2", "3"],
    [
        fl.EulerDensity,
        fl.ElerianDensity,
        fl.KesslerDensity,
        fl.DrozdovDensity,
    ],
):
    estimator = fl.KramersMoyalEstimator(model)
    res = estimator.fit_fetch(data)
    print(name, res.coefficients)
    axs[0].plot(xfa, res.drift(xfa.reshape(-1, 1)), marker, label=name)
    axs[1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), marker, label=name)
    axs[2].plot(xfa, fl.analysis.free_energy_profile_1d(res, xfa), marker, label=name)


hist, bins = np.histogram(np.concatenate([trj["x"][25000:, 0] for trj in data]), bins=50)
x_bins = 0.5 * (bins[1:] + bins[:-1])

fe_hist = -np.log(hist)

axs[2].plot(x_bins, fe_hist - np.min(fe_hist), label="Histogram")

kde = gaussian_kde(np.concatenate([trj["x"][25000::100, 0] for trj in data]))

fe_kde = -kde.logpdf(x_range)
axs[2].plot(x_range, fe_kde - np.min(fe_kde), label="KDE")


axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()
