import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay


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


def reduce_data_size_2d(X, bins=10, state_level=0.0):

    _, dim = X.shape
    if dim != 2:
        raise ValueError("Only apply to 2d data")
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    inds = np.nonzero(H > state_level)
    X = np.column_stack((xcenters[inds[0]], ycenters[inds[1]]))
    bbox = [X_min[0], X_min[1], X_max[0], X_max[1]]
    return X, bbox


def reduce_data_size_3d(X, bins=10, state_level=0.0):

    _, dim = X.shape
    if dim != 3:
        raise ValueError("Only apply to 3d data")
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    H, xedges, yedges = np.histogramnd(X[:, 0], bins=bins)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    inds = np.nonzero(H > state_level)
    X = np.column_stack((xcenters[inds[0]], ycenters[inds[1]]))
    bbox = [X_min[0], X_max[0], X_min[1], X_max[1]]
    return X, bbox


def generate_density_based_mesh(X, bins=10, state_level=1.0, k_max=4, metric="minkowski", Ninit_vertices=1000, pfix=[], alpha=1.5):
    r"""
    Density based mesh. get mesh as an union of ball arounf random points with edge length related to local densiy of points

    Parameters
    ------------

        X is the set of points within the state

        state_level is the minimal number of points per bins in the reduction

        N_max is the max number of points to be considered

        kmax is the number of neighbours taken for construction of the spanning graph

        alpha, strenght if the mesh size dependance in local density of points. alpha=0 is no dependance
    """

    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
    from distmesh import distmesh2d, distmeshnd

    dim = X.shape[1]

    X, bbox = reduce_data_size(X, bins, state_level)
    state_nbrs = NearestNeighbors(n_neighbors=k_max, algorithm="ball_tree", metric=metric).fit(X)

    # TODO : Check connected components
    connectivity_graph = state_nbrs.kneighbors_graph(mode="distance")
    n_comps, labels = connected_components(connectivity_graph)
    if n_comps > 1:
        print("WARNING there is {} connected components, increase k_max or split the set of points".format(n_comps))

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

    def hdensity(x):  # Rescaler ça pour ne pas avoir de diminution trop brusque (pas plus de 1.2*0.5)
        if x.size == 0:
            return x[:, 0:1]
        dist, _ = state_nbrs.kneighbors(x)
        res = np.power(dist[:, k_max - 1 : k_max], alpha)
        return res / np.sqrt(np.mean(res ** 2))  # Normalize in order to change only relative size

    if dim == 2:
        h0 = np.sqrt(np.abs(bbox[0] - bbox[1]) * np.abs(bbox[2] - bbox[3]) * 0.5 * np.sqrt(3) / Ninit_vertices)
        print(bbox, h0)
        pts, tri = distmesh2d(dfunc, hdensity, h0, bbox, pfix)
    else:
        h0 = np.sqrt(np.abs(bbox[0] - bbox[1]) * np.abs(bbox[2] - bbox[3]) * 0.5 * np.sqrt(3) / Ninit_vertices)
        print(bbox, h0)
        pts, tri = distmeshnd(dfunc, hdensity, h0, bbox, pfix)
    return pts, tri, X, dfunc


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
