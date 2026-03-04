from ..data import stats_from_input_data
from .._numpy import np
import skfem
from .element_finder import get_element_finder

# TODO: Add gestion of the periodicity


def are_compatible_domains(*doms):
    """
    Check whatever the listed domains are compatible.
    """
    # Check equal dimension
    if not all(d.dim == doms[0].dims for d in doms):
        return False
    # TODO: check is cube limit are the same
    return True


class Domain:
    """
    A class that represent the domain of the functions/models evaluation
    It has 2 attribute, its dimension and a cube that enclose its domain
    """

    def __init__(self, cube):
        if cube.ndim == 1:
            dim = 1
        else:
            dim = cube.shape[1]
        self.dim = dim
        self.cube = cube  # Enclosing cube

    # def localize_data(self, X):
    #     """
    #     Find cells indices and local value
    #     """
    #     return 0, None

    @classmethod
    def create_from_data(cls, data):
        stats = stats_from_input_data(data)
        range = np.empty((2, stats.dim))
        range[0, :] = stats.min
        range[1, :] = stats.max
        return cls(cube=range)

    @classmethod
    def Rd(cls, dim):
        """
        Build d dimensionnal space
        """
        assert dim >= 0
        if dim < 1:
            dim = 1
        range = np.empty((2, dim))
        range[0, :] = -np.inf
        range[1, :] = np.inf
        return cls(range)

    @classmethod
    def Td(cls, dim, min_val=0, max_val=2 * np.pi):
        """
        Build d dimensionnal torus
        """
        range = np.empty((2, dim))
        range[0, :] = min_val
        range[1, :] = max_val
        return cls(range)

    def grid(self, Npoints):
        """
        Get a grid over the domain cube, useful for plotting
        """
        return np.linspace(self.cube[0, ...], self.cube[1, ...], Npoints)


class MeshedDomain(Domain):
    def __init__(self, mesh):
        """
        Build a meshed domain from a mesh
        """
        self.mesh = mesh
        self.dim = self.mesh.dim()
        self.cube = np.linspace(np.min(self.mesh.p.T, axis=0), np.max(self.mesh.p.T, axis=0), 2)  # We get the number of points for the reference domain

    def grid(self, refined=1):
        mplot = self.mesh.refined(refined)
        return mplot.p.T

    def localize_data(self, X, mapping=None):
        """
        Get elements and position within the elements
        """
        if mapping is None:
            mapping = self.mesh.mapping()
        cells = get_element_finder(self.mesh, mapping=mapping)(*(X.T))
        loc_x = mapping.invF(X.T[:, :, np.newaxis], tind=cells)
        return cells, loc_x[..., 0].T

    def meshed_histogram(self, X, mapping=None):
        if mapping is None:
            mapping = self.mesh.mapping()
        cells = get_element_finder(self.mesh, mapping=mapping)(*(X.T))
        return cells

    @classmethod
    def create_from_range(cls, *xis, periodic=None):
        """
        Return MeshedDomain initialized from the range
        If xis is an ndarray, assume than the dimension is the smallest dim between xis.shape[0] and xis.shape[1]
        """
        if len(xis) == 1 and isinstance(xis[0], np.ndarray):
            if xis[0].ndim == 2:
                if xis[0].shape[0] > xis[0].shape[1]:
                    xis = xis[0].T.tolist()
                else:
                    xis = xis[0].tolist()
            else:
                xis = [xis[0].ravel()]
        if periodic is None:
            if len(xis) == 1:
                meshcls = skfem.MeshLine
            elif len(xis) == 2:
                meshcls = skfem.MeshTri
            elif len(xis) == 3:
                meshcls = skfem.MeshTet1
            else:
                raise ValueError("Too many inputs. Cannot create mesh of dimension higher than 3")
            return cls(meshcls.init_tensor(*xis))
        else:
            if len(xis) == 1:
                meshcls = skfem.MeshLine1DG
            elif len(xis) == 2:
                meshcls = skfem.MeshTri1DG
            else:
                raise ValueError("Too many inputs. Cannot create periodic mesh of dimension higher than 2")

            return cls(meshcls.init_tensor(*xis, periodic=periodic))

    @classmethod
    def create_from_data_with_constraints(cls, data, min_points=10, max_points=None, max_iterations=50, boundary_buffer=0.05, verbose=False):
        """
        Create a 2D MeshedDomain ensuring minimum data points per element.

        This factory method creates an unstructured triangular mesh from scattered
        data points, ensuring that each mesh element contains at least `min_points`
        data points. This is particularly useful for finite element function
        estimation where having sufficient data per element is crucial for
        accurate coefficient estimation.

        Parameters
        ----------
        data : array_like, shape (N, 2)
            Scattered data points to be distributed across the mesh.
        min_points : int, optional
            Minimum number of data points required in each element. Default is 10.
        max_points : int, optional
            Maximum number of data points allowed in each element. If None,
            no upper limit is enforced. Default is None.
        max_iterations : int, optional
            Maximum number of refinement iterations. Default is 50.
        boundary_buffer : float, optional
            Fractional buffer around data bounds for boundary. Default is 0.05.
        verbose : bool, optional
            If True, print progress information. Default is False.

        Returns
        -------
        MeshedDomain
            A MeshedDomain instance with the created mesh.
        point_counts : ndarray
            Number of data points in each mesh element.

        Raises
        ------
        ValueError
            If data is not a 2D array with shape (N, 2).

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.rand(10000, 2)
        >>> domain, counts = MeshedDomain.create_from_data_with_constraints(
        ...     data, min_points=20, max_points=100)
        >>> print(f"Created mesh with {len(counts)} elements")
        >>> print(f"Points per element: {counts.min()}-{counts.max()}")

        See Also
        --------
        create_minimum_point_mesh : Underlying mesh generation function.
        MeshedDomain.create_from_range : Create mesh from coordinate ranges.
        """
        vertices, simplices, point_counts = create_minimum_point_mesh(
            data, min_points=min_points, max_points=max_points, max_iterations=max_iterations, boundary_buffer=boundary_buffer, verbose=verbose
        )

        # Create skfem MeshTri from vertices and simplices
        mesh = skfem.MeshTri(vertices.T, simplices.T)

        return cls(mesh), point_counts


class MeshedDomain1D(MeshedDomain):
    @classmethod
    def create_from_data(cls, data, Npoints=15, repartition="uniform"):
        stats = stats_from_input_data(data)
        if repartition == "uniform":
            range = np.linspace(stats.min, stats.max, Npoints).ravel()
        elif repartition == "quantile":
            raise NotImplementedError  # To use np.percentile
        else:
            raise ValueError(f"Unknonw keyword {repartition} for repartition.")
        return cls(mesh=skfem.MeshLine(range))
