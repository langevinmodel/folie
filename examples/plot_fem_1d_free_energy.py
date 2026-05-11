"""
========================================
1D Double Well — FEM Free Energy & MFPT
========================================

Reconstruct free energy and mean first passage times from a fitted
overdamped Langevin model using the finite element method (FEM).

Compares FEM reconstruction with the analytical integration method
and with histogram/KDE estimates from simulated trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import folie as fl
import skfem

# ---- 1. Define the ground-truth model ----

# Double-well potential: V(x) = 0.1*x^4 - 4.5*x^2 + const
# Drift: F(x) = -D * dV/dx = -D * (0.8*x^3 - 9*x)
D = 0.1
coeff = D * np.array([0, 0, -4.5, 0, 0.1])  # V(x) coefficients
free_energy_exact = np.polynomial.Polynomial(coeff)
drift_coeff = D * np.array([-coeff[1], -2 * coeff[2], -3 * coeff[3], -4 * coeff[4]])

well_pos = np.sqrt(np.abs(2 * coeff[2] / (4 * coeff[4])))  # Position of the minimum of the free energy

drift_function = fl.functions.Polynomial(deg=3, coefficients=drift_coeff)
diff_function = fl.functions.Polynomial(deg=0, coefficients=np.array(D))

model_simu = fl.models.overdamped.Overdamped(drift_function, diffusion=diff_function)

# ---- 2. Simulate trajectories ----

dt = 5e-3
simulator = fl.simulations.Simulator(fl.simulations.EulerStepper(model_simu), dt)

ntraj = 30
q0 = np.full(ntraj, 0.0)
time_steps = 20000
data = simulator.run(time_steps, q0, save_every=1)

# ---- 3. Estimate model from data ----

n_knots = 12
domain = fl.MeshedDomain1D.create_from_data(data, Npoints=n_knots)
est_model = fl.models.OverdampedSplines1D(domain=domain)

km_estimator = fl.KramersMoyalEstimator(est_model)
res = km_estimator.fit_fetch(data)

# ---- 4. Reconstruct free energy with FEM ----

x_range = np.linspace(-7, 7, 200)
xfa = np.linspace(-7.0, 7.0, 150)
n_elem = 50
# FEM free energy using skfem directly
mesh = skfem.MeshLine(np.linspace(-7, 7, n_elem))
fem_basis = skfem.CellBasis(mesh, skfem.ElementLineP1())
U_coeff = fl.analysis.free_energy_profile(res, fem_basis)
V_fem = fem_basis.probes(xfa.reshape(-1, 1).T) @ U_coeff

# Analytical integration of the 1D free energy
V_analytical = fl.analysis.free_energy_profile_1d(res, xfa)

# Ground-truth free energy
V_exact = free_energy_exact(xfa)
V_exact -= np.min(V_exact)

# Histogram / KDE from trajectories
traj_concat = np.concatenate([trj["x"][5000::10, 0] for trj in data])
hist, bins = np.histogram(traj_concat, bins=80, range=(-7, 7), density=True)
x_bins = 0.5 * (bins[1:] + bins[:-1])
V_hist = -np.log(hist + 1e-12)
V_hist -= np.min(V_hist)

from scipy.stats import gaussian_kde

kde = gaussian_kde(traj_concat)
V_kde = -kde.logpdf(x_range)
V_kde -= np.min(V_kde)

# ---- 5. Compute MFPT via FEM ----

mesh_mfpt = mesh.with_subdomains({"reactant": lambda X: X[0] < -well_pos, "product": lambda X: X[0] > well_pos})
u_mfpt, states, mfpt_basis = fl.analysis.fem.solve_mfpt_fem(res, mesh_mfpt, skfem.ElementLineP1, bc="elements")
mfpt_fem_nodal = u_mfpt[mfpt_basis.nodal_dofs]
mfpt_fem = mfpt_basis.probes(xfa.reshape(1, -1)) @ u_mfpt
# Exact MFPT from simulation model
x_mfpt_exact, mfpt_exact = fl.analysis.mfpt_1d(model_simu, well_pos, [-7, 7], Npoints=2500)
# Exact MFPT from simulation model
x_mfpt_res, mfpt_res = fl.analysis.mfpt_1d(res, well_pos, [-7, 7], Npoints=2500)

# ---- 6. Plot ----

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Drift
axs[0, 0].set_title("Drift F(x)")
axs[0, 0].set_xlabel("$x$")
axs[0, 0].set_ylabel("$F(x)$")
axs[0, 0].grid()
axs[0, 0].plot(xfa, model_simu.drift(xfa.reshape(-1, 1)), label="Exact")
axs[0, 0].plot(xfa, res.drift(xfa.reshape(-1, 1)), "--", label="Estimated")
axs[0, 0].legend()

# Diffusion
axs[0, 1].set_title("Diffusion D(x)")
axs[0, 1].set_xlabel("$x$")
axs[0, 1].set_ylabel("$D(x)$")
axs[0, 1].grid()
axs[0, 1].plot(xfa, model_simu.diffusion(xfa.reshape(-1, 1)), label="Exact")
axs[0, 1].plot(xfa, res.diffusion(xfa.reshape(-1, 1)), "--", label="Estimated")
axs[0, 1].legend()

# Free energy
axs[1, 0].set_title("Free Energy V(x)")
axs[1, 0].set_xlabel("$x$")
axs[1, 0].set_ylabel("$\\beta V(x)$")
axs[1, 0].grid()
axs[1, 0].plot(xfa, V_exact, label="Exact")
axs[1, 0].plot(xfa, V_analytical, "--", label="Analytical (integration)")
axs[1, 0].plot(xfa, V_fem - np.min(V_fem), "-.", label="FEM reconstruction")
axs[1, 0].plot(x_bins, V_hist, ":", label="Histogram")
axs[1, 0].plot(x_range, V_kde, ":", label="KDE")
axs[1, 0].legend()

# MFPT
mfpt_fem_interp = mfpt_basis.probes(x_mfpt_exact.reshape(1, -1)) @ u_mfpt
axs[1, 1].set_title("Mean First Passage Time")
axs[1, 1].set_xlabel("$x$")
axs[1, 1].set_ylabel(f"MFPT to x={well_pos:.2f}")
axs[1, 1].grid()
axs[1, 1].plot(x_mfpt_exact, mfpt_exact, label="Exact model")
axs[1, 1].plot(x_mfpt_res, mfpt_res, label="Integration from estimated model")
axs[1, 1].plot(x_mfpt_exact, mfpt_fem_interp, "--", label="FEM estimated")
axs[1, 1].axvline(0, color="gray", ls=":")
axs[1, 1].axvline(-well_pos, color="gray", ls=":", alpha=0.5)
axs[1, 1].axvline(well_pos, color="gray", ls=":", alpha=0.5)
axs[1, 1].legend()

plt.tight_layout()
plt.show()
