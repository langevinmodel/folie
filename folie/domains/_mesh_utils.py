import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay
import skfem


def non_uniform_line(x_start, x_end, num_elements, ratio):
    r"""
    Creates a 1D non-uniform grid by placing num_elements nodes in
    geometric progression between x_start and x_end with ratio ratio.

    Args:
        x_start: This is the first param.
        x_end: This is a second param.
    """

    # Create grid points between 0 and 1
    h = (ratio - 1) / (ratio ** num_elements - 1)
    x = np.append([0], h * np.cumsum(ratio ** np.arange(num_elements)))

    return x_start + x * (x_end - x_start)


def data_driven_line(data, bins=10, x_start=None, x_end=None):
    r"""
    Creates a 1D grid using histogram estimation.

    Args:
        data: Data points
        x_start: Leftmost x-coordinate of the domain.
        x_end: Rightmost x-coordinate of the domain.

    Returns:
        A fully initialized instance of Mesh.
    """
    if x_start is None:
        x_start = data.min()
    if x_end is None:
        x_end = data.max()
    return np.histogram_bin_edges(data, bins=bins, range=(x_start, x_end))


def centroid_driven_line(data, bins=100):
    r"""
    Creates a mesh line based on centroids of the data, to get more cell around point with more datas

    Args:
        data: Data points, we already assume that we only have reactive trajectory
        bins: wanted number of element
    """
    kmeans = KMeans(n_clusters=bins, random_state=0).fit(data)
    # For 1D, we ravel and sort the cluster center
    # We have to add the boundary
    mesh = np.concatenate((np.array([data.min()]), np.sort(kmeans.cluster_centers_.ravel()), np.array([data.max()])))
    return mesh


def get_intersect(a1, a2, b1, b2):
    r"""
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float("inf"), float("inf"))
    return (x / z, y / z)


def remove_one_point(boundary_vertices, ratio_limit=0.01, verbose=False):
    r"""
    Remove one points from the smallest
    """
    length = np.power(boundary_vertices[1:] - boundary_vertices[:-1], 2).sum(axis=1)
    min_arete = np.argmin(length)
    ratio = length[min_arete] / np.sum(length)
    if verbose:
        print("Ratio of smallest edge to perimeter", ratio)
    if ratio < ratio_limit:
        pre_a = boundary_vertices[(min_arete - 1) % len(boundary_vertices)]
        pre_b = boundary_vertices[(min_arete) % len(boundary_vertices)]

        post_a = boundary_vertices[(min_arete + 1) % len(boundary_vertices)]
        post_b = boundary_vertices[(min_arete + 2) % len(boundary_vertices)]

        boundary_vertices[min_arete, :] = get_intersect(pre_a, pre_b, post_a, post_b)
        return np.delete(boundary_vertices, (min_arete + 1) % len(boundary_vertices), axis=0), True
    else:
        return boundary_vertices, False


def centroid_driven_mesh(data, bins=100, boundary_vertices=None, simplify_hull=0.01, verbose=False):
    r"""
    Creates a mesh line based on centroids of the data, to get more cell around point with more datas

    Args:
        data: Data points, we already assume that we only have reactive trajectory
        bins: wanted number of element
    """
    # Find clusters
    kmeans = KMeans(n_clusters=bins, random_state=0).fit(data)
    # We have to add the boundary, let's take the convex hull of the data if not defined
    if boundary_vertices is None:
        hull = ConvexHull(data)
        boundary_vertices = data[hull.vertices]
        stop = simplify_hull > 0.0
        while stop:
            boundary_vertices, stop = remove_one_point(boundary_vertices, ratio_limit=simplify_hull, verbose=verbose)
    vertices = np.concatenate((boundary_vertices, kmeans.cluster_centers_))
    # For ND, do Delaunay triangulation of the space
    tri = Delaunay(vertices)
    return vertices, tri.simplices


def reduce_data_size_support(X, bins=10, N_min=0):

    Ndata, dim = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    if dim == 1:
        H, xedges = np.histogram(X[:, 0], bins=bins)
        xcenters = [(xedges[:-1] + xedges[1:]) / 2]
    else:
        H, edges = np.histogramdd(X, bins=bins)
        xcenters = [(xedges[:-1] + xedges[1:]) / 2 for xedges in edges]

    inds = np.nonzero(H > N_min)
    H = H[H > N_min].ravel()

    X = np.column_stack([xc[ind] for xc, ind in zip(xcenters, inds)])
    bbox = np.array([*X_min, *X_max])

    return X, bbox


def mesh_on_data_support(X, bins=10, state_level=0.0, metric="minkowski", Ninit_vertices=1000, pfix=[], return_support_function=False):
    r"""
    Give uniform mesh on the support of data
    This use the distmesh algorithm :footcite:`Per-Olof Persson`

    Parameters
    ------------

        X is the set of points within the state

        state_level is the minimal number of points per bins in the reduction

        bins is the discretization used to reduce the number of data points


        Ninit_vertices is the max number of vertices in the mesh.
            Final number of mesh ertices depend of the shape of the data and maximum number of vertices should be obtained for uniform data on a rectangle

        alpha, strenght of the mesh size dependance in local density of points. alpha=0 is uniform dsitribution of the mesh
            negative alpha put more point in zone of low density and positive alpha put more points in zone of high density.
            When alpha is 1.0, the local density of mesh point should be equal to the histogram of the data

    References
    --------------

    .. footbibliography::
    """

    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
    from .distmesh import distmesh2d, distmeshnd, huniform

    dim = X.shape[1]

    if dim == 1:
        pts = np.linspace(X.min(), X.max(), Ninit_vertices)
        tri = None
    else:
        X, bbox = reduce_data_size_support(X, bins, state_level)
        k_max = 2 * dim
        state_nbrs = NearestNeighbors(n_neighbors=k_max, algorithm="ball_tree", metric=metric).fit(X)

        connectivity_graph = state_nbrs.kneighbors_graph(mode="distance")
        n_comps, labels = connected_components(connectivity_graph)
        if n_comps > 1:
            print("WARNING there is {} connected componentss".format(n_comps))
            # TODO: Faire quelque choose pour ne garder que la composante l plus grande

        spanning_tree = minimum_spanning_tree(connectivity_graph)
        state_radius = (spanning_tree + spanning_tree.T).max(axis=1).toarray()[:, 0]  # Find radius in order to get connected graph

        def dfunc(x):
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.size == 0:
                return x[:, 0]
            dist, inds = state_nbrs.kneighbors(x)
            # print(dist)
            d = dist[:, :k_max] - state_radius[inds[:, :k_max]]  # Remove distance to point
            return d.min(axis=1)

        # # L'inverse de la mesure c'est le volume du tedraedre local, donc l'arete c'est **(1/dim) de çaz
        # def hdensity(x):  # Rescaler ça pour ne pas avoir de diminution trop brusque (pas plus de 1.2*0.5)
        #     if x.size == 0:
        #         return x[:, 0:1]
        #     dists, inds = state_nbrs.kneighbors(x)
        #     # Check for zero distance, if we have a zero distances, then its weighjs is set to 1.0 and the other one to zero
        #     weights = np.empty_like(dists)
        #     with np.errstate(divide="ignore"):
        #         weights = 1.0 / dists
        #     inf_mask = np.isinf(weights)
        #     inf_row = np.any(inf_mask, axis=1)
        #     weights[inf_row] = inf_mask[inf_row]
        #     loc_density = (w[inds] * weights).sum(axis=1) / weights.sum(axis=1)  # That should give estimate of local density
        #     return 1 / (loc_density**alpha)  # That should give inverse of local density

        pfix = np.asarray(pfix).reshape(-1, dim)

        # Initial number of points estimation

        # Create uniform grid
        h0 = (np.prod([np.abs(bbox[dim + n] - bbox[n]) for n in range(dim)]) / Ninit_vertices) ** 1 / dim
        p = np.mgrid[tuple(slice(bbox[n], bbox[dim + n] + h0, h0) for n in range(dim))]
        p = p.reshape(dim, -1).T
        N_uni = p.shape[0]
        # 2. Remove points outside the region, apply the rejection method
        p = p[dfunc(p) < 0.0]  # Keep only d<0 points
        Ninit_vertices = Ninit_vertices * N_uni / p.shape[0]
        if dim == 2:
            h0 = np.sqrt(np.abs(bbox[0] - bbox[2]) * np.abs(bbox[1] - bbox[3]) * 2 / np.sqrt(3) / Ninit_vertices)
            pts, tri = distmesh2d(dfunc, huniform, h0, bbox, pfix)
        else:
            h0 = (np.prod([np.abs(bbox[dim + n] - bbox[n]) for n in range(dim)]) / Ninit_vertices) ** 1 / dim
            print(Ninit_vertices, h0)
            pts, tri = distmeshnd(dfunc, huniform, h0, bbox, pfix, fig=None)
    if return_support_function:

        return pts, tri, X, dfunc
    else:
        return pts, tri


def reduce_data_size(X, bins=10, N_min=0, N_min_per_bins=20, Ninit_vertices=1000):

    Ndata, dim = X.shape
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    if dim == 1:
        H, xedges = np.histogram(X[:, 0], bins=bins, density=True)
        xcenters = [(xedges[:-1] + xedges[1:]) / 2]
    else:
        H, edges = np.histogramdd(X, bins=bins, density=True)
        xcenters = [(xedges[:-1] + xedges[1:]) / 2 for xedges in edges]

    dx = [np.diff(xc).mean() for xc in xcenters]  # Average distances between 2 centers for all directions

    inds = np.nonzero(H > N_min)
    H = H[H > N_min].ravel()

    X = np.column_stack([xc[ind] for xc, ind in zip(xcenters, inds)])
    bbox = np.array([*X_min, *X_max])

    # Find max cap on H to avoid overconcentration of traingle on max points
    H_sorted = np.sort(H.ravel())
    Area_tot = np.prod([np.abs(bbox[dim + n] - bbox[n]) for n in range(dim)])
    k = np.argmax(np.arange(H_sorted.shape[0])[::-1] * H_sorted + np.cumsum(H_sorted) > N_min_per_bins * Ninit_vertices ** 2 / (Area_tot * Ndata))
    cap_on_H = H_sorted[k]
    H = np.minimum(H, cap_on_H)

    return X, bbox, H, dx


def generate_density_based_mesh(X, bins=10, state_level=0.0, metric="minkowski", Ninit_vertices=1000, pfix=[], alpha=1.0, return_scaling=False):
    r"""
    Density based mesh. get mesh as an union of ball arounf random points with edge length related to local densiy of points
    This use the distmesh algorithm :footcite:`Per-Olof Persson`

    Parameters
    ------------

        X is the set of points within the state

        state_level is the minimal number of points per bins in the reduction

        bins is the discretization used to reduce the number of data points


        Ninit_vertices is the max number of vertices in the mesh.
            Final number of mesh ertices depend of the shape of the data and maximum number of vertices should be obtained for uniform data on a rectangle

        alpha, strenght of the mesh size dependance in local density of points. alpha=0 is uniform dsitribution of the mesh
            negative alpha put more point in zone of low density and positive alpha put more points in zone of high density.
            When alpha is 1.0, the local density of mesh point should be equal to the histogram of the data

    References
    --------------

    .. footbibliography::
    """

    from scipy.spatial import cKDTree
    from scipy.integrate import cumulative_trapezoid
    from .densitymesh import densmesh2d

    dim = X.shape[1]

    X, bbox, w, dx = reduce_data_size(X, bins, state_level)

    if dim == 1:  # A adapter
        dhfun = w ** alpha

        hdensity = np.concatenate(([dhfun[0]], dhfun, [dhfun[-1]]))
        X = np.concatenate(([bbox[0]], X.ravel(), [bbox[1]]))
        h_scaled = cumulative_trapezoid(hdensity, X, initial=0)
        h_scaled /= h_scaled[-1]
        pts = np.interp(np.linspace(0, 1, Ninit_vertices), h_scaled, X)
        tri = None
    else:
        tree = cKDTree(X)  # Un KDTree pour sélectionner efficacement un sous ensemble des points sur lequel faire la regression
        bandwidth = np.linalg.norm(dx)

        def density(x):
            d, inds = tree.query(x, k=8)
            Kw = (1 / np.sqrt(2 * np.pi)) * w[inds] * np.exp(-0.5 * (d / bandwidth) ** 2)
            return 1e-2 - Kw.sum(axis=1)

        def grad_log_density(x):
            d, inds = tree.query(x, k=8)
            Kw = (1 / np.sqrt(2 * np.pi)) * w[inds] * np.exp(-0.5 * (d / bandwidth) ** 2)
            norm = Kw.sum(axis=1)
            num = (Kw[..., None] * (x[:, None, :] - tree.data[inds])).sum(axis=1)
            return -2 * np.divide(num, norm[:, None], out=np.zeros_like(x), where=norm[:, None] != 0)  # This is the local average of y

        pfix = np.asarray(pfix).reshape(-1, dim)

        rng = np.random.default_rng()
        u = rng.uniform(0, 1, size=Ninit_vertices)  # Selection a random gaussian
        cumsum_weight = np.cumsum(np.asarray(w))
        sum_weight = cumsum_weight[-1]
        i = np.searchsorted(cumsum_weight, u * sum_weight)

        inits_points = np.atleast_2d(rng.normal(tree.data[i], bandwidth))
        if dim == 2:
            pts, tri = densmesh2d(inits_points, grad_log_density, density, pfix)
        else:
            h0 = (np.prod([np.abs(bbox[dim + n] - bbox[n]) for n in range(dim)]) / Ninit_vertices) ** 1 / dim
            print(bbox, h0)
            pts, tri = distmeshnd(dfunc, hdensity, h0, bbox, pfix)
    if return_scaling:
        return pts, tri, X, grad_log_density(X)
    else:
        return pts, tri


def distmesh2D(fd, fh, h0, bbox, pfix, **kwargs):
    """
    Generate 2D Mesh from a distance function.
    Use pydistmesh directly but this is here for documentation purpose

    Parameters
    -------------
        fd: callable
            Distance function d([x,y]). Point are inside the mesh if the distance function is negative
        fh: callable
            Scaled edge length function h(x,y). Allow to modulate local mesh size
        h0: float
            Initial edge length
        bbox: array
            Bounding box [xmin,xmax,ymin,ymax]
        pfix:
            Fixed node positions (NFIXx2). Position of fixed nodes.

    Output:
        pts: ndarray  (Nx2)
            Node positions
        tri: ndarray  (NTx3)
            Triangle indices

    For example, the generation of a rectange  with with elliptic hole
    :: code-block: python
        import distmesh
        x1, x2, y1, y2 = 0.0, 1.0, -1.0, 1.0
        def dfunc(p):
            d0 = distmesh.drectangle(p, x1, x2, y1, y2)
            dA = distmesh.dellipse(p, xa, ya, rx, ry)
            dB = distmesh.dellipse(p, xb, yb, rx, ry)
            d = distmesh.ddiff(d0, dunion(dA, dB))
            return d

        # h0 is the desired scaling parameter for the mesh
        h0 = 0.04
        Nfix = Na + Nb + Nouter
        # bbox = [xmin,xmax,ymin,ymax]
        bbox = [xmin, xmax, ymin, ymax]
        pts, tri = distmesh.distmesh2d(dfunc, huniform, h0, bbox, [])
        mesh = meshio.Mesh(pts, [("triangle", tri)])
        print(mesh)
        mesh.write("mesh.vtk")
    """
    import distmesh

    return distmesh.distmesh2d(fd, fh, h0, bbox, pfix)


if __name__ == "__main__":  # pragma: no cover
    import matplotlib.pyplot as plt

    points = np.random.rand(5000, 2)
    vertices, tri = centroid_driven_mesh(points, 10, boundary_vertices=[[0, 0], [0, 1], [1, 0], [1, 1]])
    plt.plot(points[:, 0], points[:, 1], "x")
    plt.triplot(vertices[:, 0], vertices[:, 1], tri)
    plt.plot(vertices[:, 0], vertices[:, 1], "o")
    plt.show()
