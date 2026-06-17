"""
Set of analysis methods using Finite Element Method to solve various equations.
"""

from collections import namedtuple

import numpy as np
import skfem
from skfem import BilinearForm
from skfem.helpers import dot, mul, inv, grad
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs

# Named return types for FEM analysis functions

MfptResult = namedtuple("MfptResult", ["states", "to_product", "to_reactant", "mfpt"])
CommittorResult = namedtuple("CommittorResult", ["committor", "boundary"])
RatesResult = namedtuple("RatesResult", ["eigenvalues", "reduced_matrix", "memberships"])


class LangevinBilinearForm:
    """A class to compute value of bilinear form."""

    __name__ = "Langevin"

    def __init__(self, model, log_measure=None, verbose=False):
        self.model = model
        self.dim = model.dim
        self.log_measure = log_measure
        if verbose:
            print("{} generator of dimension {}".format(self.__name__, self.dim))

    def grammian(self, u, v, w):
        if self.log_measure is not None:
            X = w["x"].reshape(w["x"].shape[0], -1).T
            mx = np.exp(-self.log_measure(X)).reshape(w["x"].shape[1:])
            u = u * mx
        return u * v


class LangevinOverdamped(LangevinBilinearForm):
    __name__ = "LangevinOverdamped"

    def __call__(self, u, v, w):
        """Return generator for overdamped Langevin equation."""
        from skfem import BilinearForm
        from skfem.helpers import dot, mul, inv, grad

        X = w["x"].reshape(w["x"].shape[0], -1).T
        D = self.model.diffusion(X).T.reshape(w["x"].shape[0], w["x"].shape[0], *w["x"].shape[1:])
        F = self.model.pos_drift(X).T.reshape(*w["x"].shape)
        if self.log_measure is not None:
            logmx = self.log_measure(X)
            mx = np.exp(-logmx).reshape((1, *w["x"].shape[1:]))
            grad_log_m = self.log_measure.grad_x(X).T.reshape(*w["x"].shape)
            F = mx * (F + mul(D, grad_log_m))
            D = mx[None, ...] * D
        return -1 * (dot(grad(v), mul(D, grad(u))) - v * dot(F, grad(u)))

    def reversible_form(self, u, v, w):
        """Return generator for overdamped Langevin equation with Gibbs measure."""
        from skfem.helpers import dot, mul

        X = w["x"].reshape(w["x"].shape[0], -1).T
        D = self.model.diffusion(X).T.reshape(w["x"].shape[0], w["x"].shape[0], *w["x"].shape[1:])
        if self.log_measure is not None:
            logmx = self.log_measure(X)
            mx = np.exp(-logmx).reshape((1, *w["x"].shape[1:]))
            D = mx[None, ...] * D
        return -1 * dot(grad(v), mul(D, grad(u)))


class FemAnalysis:
    """
    Base class for FEM-based analysis of Langevin models.

    Provides common infrastructure: mesh, basis, matrix assembly, and PCCA++.
    Subclasses specialize in free energy or rate analysis.

    Parameters
    ----------
    model : fitted model
        A fitted folie model (e.g., Overdamped with FiniteElement drift).
    domain : MeshedDomain
        The domain with an underlying skfem mesh.
    element : skfem element or type, optional
        The finite element type. If None, uses the mesh's default element.
    """

    def __init__(self, model, domain, element=None, weight_func=None):
        self.model = model
        self.domain = domain
        self.mesh = domain.mesh
        self.element = element
        self._basis = None
        self.weight_func = weight_func  # For weighted assembly

    @property
    def basis(self):
        """Lazy-initialized skfem CellBasis."""
        if self._basis is None:
            elem = self.element if self.element else self.mesh.elem
            if isinstance(elem, type):
                elem = elem()
            self._basis = skfem.CellBasis(self.mesh, elem)
        return self._basis

    def build_matrices(self, log_measure=None):
        """
        Construct the stiffness (A) and mass (M) matrices for the FEM analysis.

        Parameters
        ----------
        log_measure : callable, optional
            Log of the reference measure for Gibbs ensemble.

        Returns
        -------
        A, M, basis : sparse matrix, sparse matrix, skfem CellBasis
        """
        from skfem import BilinearForm

        langevinform = LangevinOverdamped(self.model, log_measure)
        # if isinstance(self.model, skfem.BilinearForm):

        A = skfem.asm(BilinearForm(langevinform), self.basis)
        M = skfem.asm(BilinearForm(langevinform.grammian), self.basis)
        return A, M, self.basis


class FreeEnergyAnalysis(FemAnalysis):
    """
    Free energy (PMF) analysis using FEM.

    Computes the free energy profile from the model's force and diffusion,
    evaluates it at arbitrary points, and finds local minima.

    Parameters
    ----------
    model : fitted model
        A fitted folie model (e.g., Overdamped with FiniteElement drift).
    domain : MeshedDomain
        The domain with an underlying skfem mesh.
    element : skfem element or type, optional
        The finite element type. If None, uses the mesh's default element.

    Examples
    --------
    >>> fem = FreeEnergyAnalysis(model, domain, skfem.ElementTriP2())
    >>> Vpmf = fem.free_energy_profile()
    >>> minima = fem.find_minima(initial_centers=[np.array([0, 0]), np.array([1, 1])])
    >>> values = fem.evaluate(test_points)
    """

    def __init__(self, model, domain, **kwargs):
        super().__init__(model, domain, **kwargs)
        self._U_coeff = None

    def _build_bterm(self, v_basis=None):
        """Build the linear form for PMF computation."""
        if v_basis is None:
            v_basis = self.basis

        @skfem.LinearForm
        def b_term(v, w):
            X = w["x"].reshape(w["x"].shape[0], -1).T
            F = self.model.pos_drift(X).T.reshape(w["x"].shape)
            D = self.model.diffusion(X).T.reshape((w["x"].shape[0], w["x"].shape[0], *w["x"].shape[1:]))
            if D.shape[0] == 1:  # 1D case that is not taken into account by skfem
                D_inv = 1.0 / D
            else:
                D_inv = inv(D)

            grad_D = self.model.diffusion.grad_x(X).T.reshape((w["x"].shape[0], w["x"].shape[0], w["x"].shape[0], *w["x"].shape[1:]))
            div_D = np.einsum("jij...->i...", grad_D)

            # RHS vector field: R = D^-1 * (div_D - F)
            R = mul(D_inv, div_D - F)

            # --- Apply spatial weight ---
            weight = 1.0
            if self.weight_func is not None:
                weight = self.weight_func(X).T.reshape(w["x"].shape[1:])

            if isinstance(v_basis.elem, skfem.ElementVector):
                return weight * dot(v, R)
            else:
                return weight * dot(grad(v), R)

        return b_term

    def free_energy_profile(self, x=None, v_basis=None, alpha=0.0):
        """
        Compute the free energy profile (PMF).

        Formula: F(x) = -D(x) grad V(x) + div D(x)
        => grad V(x) = D(x)^-1 * (div D(x) - F(x))

        Parameters
        ----------
        x : ndarray or str, optional
            Evaluation points. If "nodal", return nodal values.
        v_basis : skfem Basis, optional
            Test function basis. If None, uses self.basis.
        alpha : float ≥ 0
            Tikhonov regularization strength.  alpha=0 recovers the
            original solve.  Typical useful range: 1e-5 to 1e-1.

        Returns
        -------
        ndarray
            Free energy values at evaluation points or nodal coefficients.
        """
        if v_basis is None:
            v_basis = self.basis

        b_term = self._build_bterm(v_basis)

        # --- Apply spatial weight to the Bilinear Form ---
        @skfem.BilinearForm
        def basis_product(u, v, w):
            X = w["x"].reshape(w["x"].shape[0], -1).T
            weight = 1.0
            if self.weight_func is not None:
                weight = self.weight_func(X).T.reshape(w["x"].shape[1:])

            if isinstance(v_basis.elem, skfem.ElementVector):
                return weight * dot(u.grad, v)
            else:
                return weight * dot(grad(u), grad(v))

        A = basis_product.assemble(self.basis, v_basis)
        b = b_term.assemble(v_basis)

        # Handle singularity: fix the first DOF to 0 to determine the constant shift
        self._U_coeff = skfem.solve(*skfem.condense(A, b, D=np.array([0])))
        return self

    def evaluate(self, x):
        """
        Evaluate the free energy profile at given points.

        Parameters
        ----------
        x : ndarray, shape (dim, N)
            Evaluation points.

        Returns
        -------
        ndarray, shape (N,)
            Free energy values at x.

        Raises
        ------
        RuntimeError
            If free_energy_profile() has not been called first.
        """
        if self._U_coeff is None:
            raise RuntimeError("Call free_energy_profile() first to compute the PMF coefficients.")
        if isinstance(x, str) and x == "nodal":
            return self._U_coeff[self.basis.nodal_dofs]
        return self.basis.probes(x.T) @ self._U_coeff

    def find_minima(self, initial_centers, method="Nelder-Mead", **options):
        """
        Find local minima of the free energy profile starting from given points.

        Uses the FEM basis directly to evaluate the PMF (no interpolation).

        Parameters
        ----------
        initial_centers : list of ndarray
            Starting points for each minimum search. Each element is shape (dim,).
        method : str, optional
            Optimization method. Default is "Nelder-Mead".
        **options : dict
            Additional options passed to scipy.optimize.minimize.

        Returns
        -------
        list of (position, value) tuples
            Each tuple contains the minimum position and the PMF value at that minimum.
        """
        if self._U_coeff is None:
            self.free_energy_profile()

        # Compute bounds from mesh to keep optimizer within domain
        mesh = self.basis.mesh
        bounds = []
        for dim in range(mesh.dim()):
            lo = mesh.p[dim].min()
            hi = mesh.p[dim].max()
            bounds.append((lo, hi))

        # Build an evaluator that takes a flat position and returns PMF value
        bounds_lo = [b[0] for b in bounds]
        bounds_hi = [b[1] for b in bounds]

        def objective_flat(x_flat):
            # Clamp to mesh bounds to avoid probes() errors
            x_clamped = np.clip(x_flat, bounds_lo, bounds_hi)
            X = x_clamped.reshape(1, -1).T
            try:
                val = self.basis.probes(X) @ self._U_coeff
                return val[0]
            except ValueError:
                return np.inf

        minima = []
        for x0 in initial_centers:
            result = minimize(objective_flat, x0, method=method, **options)
            if result.success:
                minima.append((result.x, result.fun))
            else:
                minima.append((x0, objective_flat(x0)))
        return minima


class RateAnalysis(FemAnalysis):
    """
    Rate analysis using FEM: MFPT, committor, and eigenmode decomposition.

    Parameters
    ----------
    model : fitted model
        A fitted folie model (e.g., Overdamped with FiniteElement drift).
    domain : MeshedDomain
        The domain with an underlying skfem mesh (must have 'reactant' and 'product' subdomains).
    element : skfem element or type, optional
        The finite element type. If None, uses the mesh's default element.

    Examples
    --------
    >>> mesh = domain.mesh.with_subdomains({"reactant": r_fn, "product": p_fn})
    >>> rate = RateAnalysis(model, MeshedDomain(mesh), skfem.ElementTriP1())
    >>> mfpt_result = rate.mfpt()
    >>> committor_result = rate.committor()
    >>> rates_result = rate.transition_rates(n_states=2)
    """

    def mfpt(self, x=None, bc="facets", mesh=None):
        """
        Compute Mean First Passage Time via FEM.

        Parameters
        ----------
        x : ndarray or str, optional
            Evaluation points. If "nodal", return nodal values.
        bc : str, optional
            Boundary condition type: "facets" or "elements".
        mesh : skfem Mesh, optional
            Alternative mesh with subdomains. If None, uses self.basis.mesh.

        Returns
        -------
        MfptResult
            Named tuple with states, to_product, to_reactant, rates.
        """
        if mesh is not None and mesh is not self.mesh:
            self.mesh = mesh
            self._basis = None  # Reset to force rebuild

        mesh = mesh or self.basis.mesh
        langevinform = LangevinOverdamped(self.model, None)
        A = skfem.asm(skfem.BilinearForm(langevinform), self.basis)

        if bc == "facets":
            product_dofs = self.basis.get_dofs({"product"})
            reactants_dofs = self.basis.get_dofs({"reactant"})
        else:
            product_dofs = np.unique(self.basis.element_dofs[:, mesh.subdomains["product"]])
            reactants_dofs = np.unique(self.basis.element_dofs[:, mesh.subdomains["reactant"]])

        @skfem.LinearForm
        def rhs(v, _):
            return -1.0 * v

        b = skfem.asm(rhs, self.basis)

        u_to_product = np.zeros(self.basis.N)
        u_to_reactant = np.zeros(self.basis.N)

        u_sol_to_product = skfem.solve(*skfem.condense(A, b, u_to_product, D=product_dofs))
        u_sol_to_reactant = skfem.solve(*skfem.condense(A, b, u_to_reactant, D=reactants_dofs))

        states = np.zeros(self.basis.N)
        states[product_dofs] = 1.0
        states[reactants_dofs] = -1.0

        # Compute transition rates from MFPT
        mfpt_vals = {}
        if np.any(u_sol_to_product[reactants_dofs] > 0):
            mfpt_vals["reactant_to_product"] = np.min(u_sol_to_product[reactants_dofs])
        if np.any(u_sol_to_reactant[product_dofs] > 0):
            mfpt_vals["product_to_reactant"] = np.min(u_sol_to_reactant[product_dofs])

        if x is not None:
            if isinstance(x, str) and x == "nodal":
                return MfptResult(states[self.basis.nodal_dofs], u_sol_to_product[self.basis.nodal_dofs], u_sol_to_reactant[self.basis.nodal_dofs], mfpt_vals)
            bp = self.basis.probes(x.T)
            return MfptResult(bp @ states, bp @ u_sol_to_product, bp @ u_sol_to_reactant, mfpt_vals)
        return MfptResult(states, u_sol_to_product, u_sol_to_reactant, mfpt_vals)

    def committor(self, x=None, bc="facets"):
        """
        Compute the committor function via FEM.

        Parameters
        ----------
        x : ndarray or str, optional
            Evaluation points. If "nodal", return nodal values.
        bc : str, optional
            Boundary condition type: "facets" or "elements".

        Returns
        -------
        CommittorResult
            Named tuple with committor values and boundary values.
        """
        langevinform = LangevinOverdamped(self.model, None)
        A = skfem.asm(skfem.BilinearForm(langevinform), self.basis)

        if bc == "facets":
            product_dofs = self.basis.get_dofs({"product"})
            reactants_dofs = self.basis.get_dofs({"reactant"})
        else:
            product_dofs = np.unique(self.basis.element_dofs[:, self.mesh.subdomains["product"]])
            reactants_dofs = np.unique(self.basis.element_dofs[:, self.mesh.subdomains["reactant"]])

        u = np.zeros(self.basis.N)
        boundary_dofs = np.concatenate((product_dofs, reactants_dofs))
        u[product_dofs] = 1.0
        u[reactants_dofs] = -1.0
        u_sol = skfem.solve(*skfem.condense(A, np.zeros_like(u), u, D=boundary_dofs))

        if x is not None:
            if isinstance(x, str) and x == "nodal":
                return CommittorResult(u_sol[self.basis.nodal_dofs], u[self.basis.nodal_dofs])
            bp = self.basis.probes(x.T)
            return CommittorResult(bp @ u_sol, bp @ u)
        return CommittorResult(u_sol, u)

    def reduced_matrix(self, L, M, n_states, verbose=True, clip_matrix=False):
        """
        Compute the reduced matrix using PCCA++.

        Parameters
        ----------
        L : sparse matrix
            Stiffness (generator) matrix.
        M : sparse matrix
            Mass matrix.
        n_states : int
            Number of metastable states.
        verbose : bool, optional
            Print eigenvalues and spectral ratio. Default is True.
        clip_matrix : bool, optional
            Clip the reduced matrix to be a valid transition matrix. Default is False.

        Returns
        -------
        L_reduced, memberships : ndarray, ndarray
        """
        from ._pcca_utils import _pcca_connected

        basis = self.basis
        eigsv, x_im = eigs(L, M=M, k=n_states + 1, sigma=0.0, which="LM")
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
        return L_reduced, memberships

    def transition_rates(self, n_states=2, log_measure=None):
        """
        Compute transition rates from eigenvalue decomposition.

        Parameters
        ----------
        n_states : int, optional
            Number of metastable states. Default is 2.
        log_measure : callable, optional
            Log of the reference measure for Gibbs ensemble.

        Returns
        -------
        RatesResult
            Named tuple with eigenvalues, reduced_matrix, memberships.
        """
        A, M, basis = self.build_matrices(log_measure)
        L_reduced, memberships = self.reduced_matrix(A, M, n_states, verbose=True, clip_matrix=True)

        # Compute eigenvalues of reduced matrix for rates
        eigenvalues = np.linalg.eigvals(L_reduced)
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]

        return RatesResult(eigenvalues, L_reduced, memberships)

    def transition_rate_from_mfpt(self, mfpt_result, bc="facets"):
        """
        Compute transition rate from MFPT values.

        Parameters
        ----------
        mfpt_result : MfptResult or ndarray
            MFPT values (can be named tuple or raw array).
        bc : str, optional
            Boundary condition type: "facets" or "elements".

        Returns
        -------
        float
            Transition rate (reactant to product).
        """
        if hasattr(mfpt_result, "to_product"):
            mfpt_values = mfpt_result.to_product[0]
        else:
            mfpt_values = mfpt_result

        if bc == "facets":
            reactants_dofs = self.basis.get_dofs({"reactant"})
        else:
            reactants_dofs = np.unique(self.basis.element_dofs[:, self.mesh.subdomains["reactant"]])
        return np.min(mfpt_values[reactants_dofs])
