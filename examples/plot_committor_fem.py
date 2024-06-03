#!python3
# -*- coding: utf-8 -*-

"""
==================================
Committor for Overdamped Langevin
==================================

How to run a simple analysis with FEM
"""
import numpy as np
import skfem
import folie as fl
from folie.analysis import solve_committor_fem


min_pot_R = [0.5, 0]
min_pot_P = [-0.5, 1.5]
state_radius = 0.05


def locate_R(x):
    return ((x - np.array(min_pot_R)[:, None]) ** 2).sum(axis=0) < state_radius ** 2


def locate_P(x):
    return ((x - np.array(min_pot_P)[:, None]) ** 2).sum(axis=0) < state_radius ** 2


potential = fl.functions.MullerBrown()

# Build model
diff_function = fl.functions.Polynomial(deg=0, coefficients=np.asarray([0.5]) * np.eye(2, 2))
model = fl.models.overdamped.Overdamped(potential, diffusion=diff_function)

# Build 2D mesh
N_points = 35
line_points_x = np.linspace(-1.5, 1.2, N_points)
line_points_y = np.linspace(-0.5, 2.0, N_points)
m = skfem.MeshTri().init_tensor(*(line_points_x, line_points_y))
m = m.with_subdomains({"reactant": locate_R, "product": locate_P})
# m = m.with_boundaries({"reactant": lambda x:((x -min_pot_R) ** 2).sum(axis=0) < state_radius ** 2, "product": lambda x:((x -min_pot_P) ** 2).sum(axis=0) < state_radius ** 2})

e = skfem.ElementTriP2()

u_sol, states_fem, basis = solve_committor_fem(model, m, e, bc="domains")

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, draw
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    pot_val = basis.project(lambda x: potential.potential(x.reshape(x.shape[0],-1).T).reshape(x.shape[1:]))
    mplot, z = basis.refinterp(pot_val, nrefs=2)
    z -= np.min(z)

    im = axs[0].tricontourf(mplot.p[0], mplot.p[1], mplot.t.T, z, levels=np.linspace(0, 60, 16), cmap="viridis")
    plt.colorbar(im, ax=axs[0], label=r"$\beta V(x,y) $")
    draw(m, ax=axs[0])

    plot(basis, (u_sol + 1) / 2, ax=axs[1], shading="gouraud", cmap="viridis", colorbar="Committor")
    axs[1].tricontour(mplot.p[0], mplot.p[1], mplot.t.T, z, levels=np.linspace(0, 60, 10), **{**{"colors": "k"}})
    circleR = plt.Circle((min_pot_R[0], min_pot_R[1]), 0.15, color="white")
    circleP = plt.Circle((min_pot_P[0], min_pot_P[1]), 0.15, color="white")
    axs[1].add_patch(circleR)
    axs[1].add_patch(circleP)
    axs[1].text(min_pot_R[0] - 0.04, min_pot_R[1] - 0.04, "R")
    axs[1].text(min_pot_P[0] - 0.04, min_pot_P[1] - 0.04, "P")

    plt.show()
