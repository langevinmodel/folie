"""
========================================
2D Muller-Brown — FEM Committor & PCCA
========================================

Compute committor probabilities and reduced Markov state kinetics
for a 2D Muller-Brown potential using FEM eigenmode decomposition and PCCA++.

Demonstrates:
- RateAnalysis.committor() for committor between two basins
- RateAnalysis.reduced_matrix() for PCCA++ coarse-graining
- Eigenmode visualization for collective variables
"""

import numpy as np
import matplotlib.pyplot as plt
import folie as fl
import skfem
from scipy.sparse.linalg import eigs

# ---- Define 2D Muller-Brown model ----

# Minima at (~1, 0), (~0, 0.5), (~-0.5, 1.5)
D = 1.0

muller_brown = fl.functions.MullerBrown()
diff_function = fl.functions.Polynomial(deg=0, coefficients=D * np.eye(2, 2), output_shape=(2, 2))

model = fl.models.overdamped.Overdamped(muller_brown, diffusion=diff_function)

# ---- Create FEM mesh and build matrices ----

# Use a mesh that covers the Muller-Brown minima region
x_min, x_max = -1.5, 1.5
y_min, y_max = -0.5, 2.0
n_x, n_y = 25, 25

x_range = np.linspace(x_min, x_max, n_x)
y_range = np.linspace(y_min, y_max, n_y)

mesh = skfem.MeshTri.init_tensor(x_range, y_range)
# ---- 4. Define subdomains (reactant and product basins) ----

# Reactant basin around global minimum (-0.5, 0), product basin around local minimum (0.5, 0)
mesh = mesh.with_subdomains(
    {
        "reactant": lambda X: (X[0] + 0.5) ** 2 + (X[1] - 1.5) ** 2 < 0.05,
        "product": lambda X: (X[0] - 0.5) ** 2 + X[1] ** 2 < 0.05,
    }
)


# Build FEM matrices
fem = fl.analysis.RateAnalysis(model, fl.MeshedDomain(mesh), element=skfem.ElementTriP1())
A, M, basis = fem.build_matrices()
print(f"FEM mesh: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} elements")
print(f"FEM basis: {basis.N} DOFs")

reactant_dofs = basis.get_dofs(elements="reactant")
product_dofs = basis.get_dofs(elements="product")
print(f"Reactant DOFs: {len(reactant_dofs)}, Product DOFs: {len(product_dofs)} (element subdomains, not facets)")

# ---- Solve committor equation ----

committor_result = fem.committor(bc="elements")
u_sol = committor_result.committor
u_bc = committor_result.boundary
fem_basis = fem.basis

# Evaluate committor on grid
n_plot = 150
x_plot = np.linspace(x_min, x_max, n_plot)
y_plot = np.linspace(y_min, y_max, n_plot)
X_grid, Y_grid = np.meshgrid(x_plot, y_plot)
X_eval = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

committor_nodal = u_sol[fem_basis.nodal_dofs]
# For tensor meshes, nodal_dofs may be empty; always use probes for 2D
committor_grid = (fem_basis.probes(X_eval.T) @ u_sol).reshape(n_plot, n_plot)

# ---- Compute eigenmodes for PCCA++ ----

n_states = 3
eigvals, eigvecs = eigs(A, M=M, k=n_states + 1, sigma=0.0, which="LM")

# Sort by real part (ascending, smallest magnitude first)
ind_sort = np.argsort(np.real(eigvals))[::-1]
eigvals = eigvals[ind_sort]
eigvecs = eigvecs[:, ind_sort]

print("\nLowest eigenvalues:")
for i in range(n_states + 1):
    print(f"  lambda_{i} = {np.real(eigvals[i]):.6f} + {np.imag(eigvals[i]):.6f}j")

# Spectral ratio
spectral_ratio = np.abs(np.real(eigvals[n_states])) / np.abs(np.real(eigvals[n_states - 1]))
print(f"\nSpectral ratio: {spectral_ratio:.4f}")

# ---- PCCA++ coarse-graining ----

L_reduced, memberships = fem.reduced_matrix(A, M, n_states, verbose=True)

# Clip and normalize reduced matrix
L_clipped = np.clip(L_reduced, 0.0, np.max(L_reduced))
for n in range(L_clipped.shape[0]):
    L_clipped[n, n] = -np.sum(L_clipped[n, :])

# Evaluate membership functions on grid (use probes for tensor meshes)
membership_grid = np.empty((n_states, n_plot, n_plot))
for s in range(n_states):
    mem_nodal = memberships[:, s][fem_basis.nodal_dofs]
    membership_grid[s] = (fem_basis.probes(X_eval.T) @ memberships[:, s]).reshape(n_plot, n_plot)

# ---- 8. Evaluate free energy on grid for comparison ----

X_eval_2d = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
V_exact = muller_brown.potential(X_eval_2d).reshape(n_plot, n_plot)
V_exact -= V_exact.min()
# ---- 9. Plot ----

fig = plt.figure(figsize=(18, 5))

# Panel 1: Free energy contour
ax1 = fig.add_subplot(2, 3, 1)
cf1 = ax1.contourf(X_grid, Y_grid, V_exact, levels=np.linspace(0, 35, 30), vmin=0, vmax=40, cmap="viridis")
ax1.contour(X_grid, Y_grid, V_exact, levels=[0, 2, 5, 10, 20, 30], colors="white", alpha=0.5, linewidths=0.5)
ax1.set_title("Free Energy V(x,y)")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_aspect("equal")
plt.colorbar(cf1, ax=ax1, shrink=0.8)

# Panel 2: Committor probability
ax2 = fig.add_subplot(2, 3, 2)
cf2 = ax2.contourf(X_grid, Y_grid, 0.5 * (1 + committor_grid), levels=50, cmap="coolwarm")  # Committor values are within -1 and 1
ax2.contour(X_grid, Y_grid, V_exact, levels=[0, 2, 5, 10, 20, 30], colors="white", alpha=0.5, linewidths=0.5)
ax2.contour(X_grid, Y_grid, 0.5 * (1 + committor_grid), levels=[0.02, 0.5, 0.98], colors="black", linewidths=1.5)
# Overlay reactant and product basin markers
reactant_nodes = mesh.p[:, reactant_dofs]
product_nodes = mesh.p[:, product_dofs]
ax2.scatter(reactant_nodes[0], reactant_nodes[1], c="white", s=15, marker="o", facecolors="none", edgecolors="white", linewidths=1.5, label="reactant")
ax2.scatter(product_nodes[0], product_nodes[1], c="white", s=15, marker="s", facecolors="none", edgecolors="white", linewidths=1.5, label="product")
ax2.legend(loc="upper right", fontsize=8)
ax2.set_title("Committor p(x)")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_aspect("equal")
plt.colorbar(cf2, ax=ax2, shrink=0.8)

# Panel 3-5: Membership functions for each state
for s in range(n_states):
    ax = fig.add_subplot(2, 3, 4 + s)
    cf = ax.contourf(X_grid, Y_grid, membership_grid[s], levels=np.linspace(0, 1, 30), cmap="viridis")
    ax.set_title(f"State {s} membership")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, shrink=0.8)

# Panel 6: Reduced transition matrix
ax6 = fig.add_subplot(2, 3, 3)
im = ax6.imshow(L_clipped, cmap="RdBu_r", vmin=-L_clipped.max(), vmax=L_clipped.max(), aspect="auto", origin="upper")
ax6.set_title(f"Reduced K\n(spectral ratio={spectral_ratio:.3f})")
ax6.set_xlabel("From state")
ax6.set_ylabel("To state")
ax6.set_xticks(range(n_states))
ax6.set_yticks(range(n_states))
# Add values as text
for i in range(n_states):
    for j in range(n_states):
        color = "white" if abs(L_clipped[i, j]) > L_clipped.max() / 3 else "black"
        ax6.text(j, i, f"{L_clipped[i, j]:.2f}", ha="center", va="center", fontsize=9, color=color)
plt.colorbar(im, ax=ax6, shrink=0.8)

plt.tight_layout()
plt.show()
