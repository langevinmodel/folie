"""
Set of analysis methods using Finite Element Method to solve various equation
"""

import numpy as np
import skfem
from skfem import BilinearForm
from skfem.helpers import grad, dot, mul
from scipy.sparse.linalg import eigs, eigsh

from ..models import BaseModelOverdamped


class LangevinBilinearForm:
    """
    A class to compute value of bilinear form
    """

    __name__ = "Langevin"

    def __init__(self, model, verbose=False):
        """"""
        self.model = model
        self.dim = model.dim
        if verbose:
            print("{} generator of dimension {}".format(self.__name__, self.dim))


class LangevinOverdamped(LangevinBilinearForm):
    __name__ = "LangevinOverdamped"

    def __call__(self, u, v, w):
        """
        Return generator for overdamped Langevin equation
        To use Gibbs measure, set force to zero and adapt the basis
        """
        # TODO: Il faut reshape comme il faut

        X = w["x"].reshape(w["x"].shape[0], -1).T
        D = self.model.diffusion(X).T.reshape(w["x"].shape[0], w["x"].shape[0], *w["x"].shape[1:])
        F = self.model.force(X).T.reshape(*w["x"].shape)
        return -1 * (dot(grad(v), mul(D, grad(u))) - v * dot(F, grad(u)))


def grammian(u, v, w):
    return u * v


def build_fem_matrices(model, mesh, element=None):
    """
    Construct the necessary matrices
    """
    if element is None:
        element = mesh.elem
    if isinstance(model, BaseModelOverdamped):
        langevinform = BilinearForm(LangevinOverdamped(model))
    basis = skfem.CellBasis(mesh, element)
    A = skfem.asm(langevinform, basis)
    M = skfem.asm(BilinearForm(grammian), basis)
    return A, M, basis


def solve_committor_fem(model, mesh, element=None, bc="facets", solver=None):
    """ """
    A, basis = build_fem_matrices(model, mesh, element)

    if bc == "facets":
        product_dofs = basis.get_dofs({"product"})
        reactants_dofs = basis.get_dofs({"reactant"})
    else:
        product_dofs = np.unique(basis.element_dofs[:, mesh.subdomains["product"]])
        reactants_dofs = np.unique(basis.element_dofs[:, mesh.subdomains["reactant"]])
    u = np.zeros(basis.N)
    boundary_dofs = np.concatenate((product_dofs, reactants_dofs))
    u[product_dofs] = 1.0
    u[reactants_dofs] = -1.0
    u_sol = skfem.solve(*skfem.condense(A, np.zeros_like(u), u, D=boundary_dofs))
    return u_sol, u, basis


def solve_mfpt_fem(model, mesh, element, solver=None, bc="facets"):
    """
    Compute FEM matrix and solve MFPT equation
    """
    A, basis = build_fem_matrices(model, mesh, element)

    if bc == "facets":
        product_dofs = basis.get_dofs({"product"})
    else:
        product_dofs = np.unique(basis.element_dofs[:, mesh.subdomains["product"]])

    @skfem.LinearForm
    def rhs(v, _):
        return -1.0 * v

    u = np.zeros(basis.N)
    states = np.zeros(basis.N)
    b = skfem.asm(rhs, basis)
    boundary_dofs = product_dofs
    u[product_dofs] = 0.0  # skfem.project(lambda x: 1, basis_to=basis, I=product_dofs)
    states[product_dofs] = 1.0
    u_sol = skfem.solve(*skfem.condense(A, b, u, D=boundary_dofs), solver=solver)  # Doitêtre égal à -1
    return u_sol, states, basis


def reduced_matrix(L, M, basis, n_states, verbose=True, clip_matrix=False):
    """
    Compute the reduced matrix using PCCA++
    """
    from _pcca_utils import _pcca_connected

    eigsv, x_im = eigs(L, M=M, k=n_states + 1, which="SM")  #
    ind_sort = np.argsort(np.real(eigsv))[::-1]
    x_left = np.real(x_im[:, ind_sort])[:, :n_states]
    eigvals = eigsv[ind_sort]
    if verbose:
        print("Eigenvalues", eigvals)
        print("Spectral ratio", np.abs(eigvals[n_states]) / np.abs(eigvals[n_states - 1]))

    eigen_vect_on_quad_point = np.empty((basis.nelems, len(basis.W), n_states))
    for n in range(n_states):
        eigen_vect_on_quad_point[..., n] = basis.interpolate(x_left[:, n])

    memberships_on_quad = _pcca_connected(eigen_vect_on_quad_point[..., 1:].reshape(-1, n_states - 1)).reshape(eigen_vect_on_quad_point.shape)
    memberships = np.empty_like(x_left)
    for n in range(n_states):
        memberships[:, n] = basis.project(memberships_on_quad[:, :, n])

    invG_membership = np.linalg.inv(memberships.T @ M @ memberships)
    L_reduced = invG_membership @ memberships.T @ L @ memberships
    if clip_matrix:
        L_reduced = np.clip(L_reduced, 0.0, np.max(L_reduced))
        for n in range(L_reduced.shape[0]):
            L_reduced[n, n] = -np.sum(L_reduced[n, :])
    return L_reduced
