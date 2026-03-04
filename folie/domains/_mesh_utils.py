import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from .distmesh import distmesh, huniform


def centroid_driven_line(data, bins=100):
    """
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
    """
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
    """
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
    """
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


def reduce_data_size(X, bins=10, N_min=0, N_min_per_bins=20, Ninit_vertices=1000):
    """
    Compact the data to have an usable representation

    :param X: Data points
    :param bins: Description
    :param N_min: Description
    :param N_min_per_bins: Description
    :param Ninit_vertices: Description
    """

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

    # Find max cap on H to avoid overconcentration of triangle on max points
    H_sorted = np.sort(H.ravel())
    Area_tot = np.prod([np.abs(bbox[dim + n] - bbox[n]) for n in range(dim)])
    k = np.argmax(np.arange(H_sorted.shape[0])[::-1] * H_sorted + np.cumsum(H_sorted) > N_min_per_bins * Ninit_vertices**2 / (Area_tot * Ndata))
    cap_on_H = H_sorted[k]
    H = np.minimum(H, cap_on_H)

    return X, bbox, H, dx


def support_based_mesh(X, bins=40, N_min=0, Ninit_vertices=1000, pfix=[], return_functions=False):
    """
    Generates a uniform mesh over the support of the data.
    Inlined data reduction and Grid-based SDF for O(1) distance queries.


    Parameters
    ------------

        X is the set of points within the state

        bins is the discretization used to reduce the number of data points. This control the length scale for support determination

        N_min is the minimal number of points per bins in the reduction



        Ninit_vertices is the max number of vertices in the mesh.
            Final number of mesh vertices depend of the shape of the data and maximum number of vertices should be obtained for uniform data on a rectangle


    """

    _, dim = X.shape
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    bbox = np.array([*X_min, *X_max])
    pfix = np.asarray(pfix).reshape(-1, dim)

    # --- 1. Handle 1D Case ---
    if dim == 1:
        pts = np.linspace(X_min[0], X_max[0], Ninit_vertices).reshape(-1, 1)
        return (pts, None, X, None, None) if return_functions else (pts, None)

    # --- 2. Inlined Reduction & Grid Masking ---
    # We use a histogram to find where the data "exists"
    H, edges = np.histogramdd(X, bins=bins)
    grid_mask = H > N_min

    # Centers of the bins for interpolation
    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    # Physical size of one bin in each dimension
    dx = [np.diff(c).mean() if len(c) > 1 else 1.0 for c in centers]

    # Bridge small gaps in data support to ensure a single connected component
    grid_mask = binary_dilation(grid_mask)

    # --- 3. Compute Signed Distance Field (SDF) ---
    # EDT computes the Euclidean distance to the nearest background pixel
    dist_inside = distance_transform_edt(grid_mask)
    dist_outside = distance_transform_edt(~grid_mask)

    # Negative inside the support, positive outside
    # Scale unit distances by physical bin size (dx)
    sdf_grid = (dist_outside - dist_inside) * np.mean(dx)

    # --- 4. Setup Distance Function (O(1) Interpolation) ---
    # This replaces the O(log N) Nearest Neighbor search
    sdf_interp = RegularGridInterpolator(centers, sdf_grid, method="linear", bounds_error=False, fill_value=np.mean(dx) * bins)

    def dfunc(p):
        return sdf_interp(p)

    # --- 5. Optimal h0 Calculation ---
    # Support Area = (number of True pixels) * (area of one pixel)
    pixel_volume = np.prod(dx)
    total_support_volume = np.sum(grid_mask) * pixel_volume

    # Target h0 for uniform triangles: Area = N * (sqrt(3)/2 * h0^2)
    # 0.866 is approx sqrt(3)/2
    h0 = (total_support_volume / (Ninit_vertices * 0.866)) ** (1.0 / dim)

    # --- 6. Run DistMesh ---
    pts, tri = distmesh(dfunc, huniform, h0, bbox, pfix=pfix)

    if return_functions:
        return pts, tri, centers, dfunc, huniform
    else:
        return pts, tri


def density_based_mesh(X, bins=40, N_min=0, Ninit_vertices=1000, pfix=[], min_scale=0.1, alpha=0.5, return_functions=False):
    """
    Generates a density-based mesh over the support of the data.

    Parameters:
    -----------
    min_scale : Finest triangle size relative to background h0 (e.g. 0.1).
    alpha : Sensitivity of mesh density to data density (0.5 to 1.0 is standard).
    """
    _, dim = X.shape
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    bbox = np.array([*X_min, *X_max])
    pfix = np.asarray(pfix).reshape(-1, dim)

    if dim == 1:
        pts = np.linspace(X_min[0], X_max[0], Ninit_vertices).reshape(-1, 1)
        return (pts, None, X, None, None) if return_functions else (pts, None)

    # --- 1. Grid Generation & Histogramming ---
    H, edges = np.histogramdd(X, bins=bins)
    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    dx = [np.diff(c).mean() if len(c) > 1 else 1.0 for c in centers]

    # --- 2. Define Support (dfunc) ---
    grid_mask = H > N_min
    grid_mask = binary_dilation(grid_mask)  # Bridge small gaps

    dist_inside = distance_transform_edt(grid_mask)
    dist_outside = distance_transform_edt(~grid_mask)
    sdf_grid = (dist_outside - dist_inside) * np.mean(dx)

    dfunc_interp = RegularGridInterpolator(centers, sdf_grid, method="linear", bounds_error=False, fill_value=np.mean(dx) * bins)

    # --- 3. Define Sizing (fh) ---
    # Smooth the histogram to ensure sizing transitions are continuous
    H_smoothed = gaussian_filter(H, sigma=1.0)

    # Rescale Density to Sizing [min_scale, 1.0]
    h_max_dens = H_smoothed.max() if H_smoothed.max() > 0 else 1.0
    # norm_d is 1.0 at max density, 0.0 at empty
    norm_d = (H_smoothed / h_max_dens) ** alpha
    # sizing_grid is min_scale at max density, 1.0 at empty
    sizing_grid = 1.0 - (1.0 - min_scale) * norm_d

    fh_interp = RegularGridInterpolator(centers, sizing_grid, method="linear", bounds_error=False, fill_value=1.0)

    # --- 4. Calculate h0 and Starting Resolution ---
    pixel_volume = np.prod(dx)
    total_support_volume = np.sum(grid_mask) * pixel_volume

    # We define h0 as the BACKGROUND (coarsest) size
    h0_background = (total_support_volume / (Ninit_vertices * 0.866)) ** (1.0 / dim)

    # IMPORTANT: To capture dense areas, DistMesh must start at the FINEST size
    h0_start = h0_background * min_scale

    # --- 5. Run DistMesh ---
    # dfunc(p) <= 0 defines the shape
    # fh(p) in [min_scale, 1.0] defines the local refinement
    pts, tri = distmesh(dfunc_interp, fh_interp, h0_start, bbox, pfix=pfix, max_iter=400)

    if return_functions:
        return pts, tri, centers, dfunc_interp, fh_interp
    else:
        return pts, tri


# def mesh_on_data_support(X, bins=10, state_level=0.0, metric="minkowski", Ninit_vertices=1000, pfix=[], return_support_function=False):
#     r"""
#     Give uniform mesh on the support of data
#     This use the distmesh algorithm :footcite:`Per-Olof Persson`

#     Parameters
#     ------------

#         X is the set of points within the state

#         state_level is the minimal number of points per bins in the reduction

#         bins is the discretization used to reduce the number of data points


#         Ninit_vertices is the max number of vertices in the mesh.
#             Final number of mesh vertices depend of the shape of the data and maximum number of vertices should be obtained for uniform data on a rectangle


#     References
#     --------------

#     .. footbibliography::
#     """

#     from sklearn.neighbors import NearestNeighbors
#     from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
#     from .distmesh import distmesh, huniform

#     dim = X.shape[1]

#     if dim == 1:
#         pts = np.linspace(X.min(), X.max(), Ninit_vertices)
#         return (pts, None, X, None) if return_support_function else (pts, None)

#     X, bbox = reduce_data_size_support(X, bins, state_level)
#     k_max = 2 * dim
#     state_nbrs = NearestNeighbors(n_neighbors=k_max, algorithm="ball_tree", metric=metric).fit(X)

#     connectivity_graph = state_nbrs.kneighbors_graph(mode="distance")
#     n_comps, labels = connected_components(connectivity_graph)
#     if n_comps > 1:
#         print("WARNING there is {} connected componentss".format(n_comps))
#         # TODO: Faire quelque choose pour ne garder que la composante l plus grande

#     spanning_tree = minimum_spanning_tree(connectivity_graph)
#     state_radius = (spanning_tree + spanning_tree.T).max(axis=1).toarray()[:, 0]  # Find radius in order to get connected graph

#     def dfunc(x):
#         x = np.asarray(x)
#         if x.ndim == 1:
#             x = x.reshape(1, -1)
#         if x.size == 0:
#             return x[:, 0]
#         dist, inds = state_nbrs.kneighbors(x)
#         # print(dist)
#         d = dist[:, :k_max] - state_radius[inds[:, :k_max]]  # Remove distance to point
#         return d.min(axis=1)

#     # L'inverse de la mesure c'est le volume du tedraedre local, donc l'arete c'est **(1/dim) de çaz

#     pfix = np.asarray(pfix).reshape(-1, dim)

#     # Initial number of points estimation

#     # Create uniform grid
#     h0 = (np.prod([np.abs(bbox[dim + n] - bbox[n]) for n in range(dim)]) / Ninit_vertices) ** (1.0 / dim)
#     p = np.mgrid[tuple(slice(bbox[n], bbox[dim + n] + h0, h0) for n in range(dim))]
#     p = p.reshape(dim, -1).T
#     N_uni = p.shape[0]
#     # 2. Remove points outside the region, apply the rejection method
#     p = p[dfunc(p) < 0.0]  # Keep only d<0 points
#     Ninit_vertices = Ninit_vertices * N_uni / p.shape[0]  # Rescale number of initial vertices by real volue of support function
#     h0 = (np.prod([np.abs(bbox[dim + n] - bbox[n]) for n in range(dim)]) / Ninit_vertices) ** (1 / dim)
#     pts, tri = distmesh(dfunc, huniform, h0, bbox, pfix)

#     if return_support_function:

#         return pts, tri, X, dfunc
#     else:
#         return pts, tri


# def generate_density_based_mesh(X, bins=10, state_level=0.0, alpha=1.0, h_min=0.1, Ninit_vertices=1000, pfix=[], return_scaling=False):
#     r"""
#     Density based mesh. get mesh as an union of ball arounf random points with edge length related to local densiy of points
#     This use the distmesh algorithm :footcite:`Per-Olof Persson`

#     Parameters
#     ------------

#         X is the set of points within the state

#         state_level is the minimal number of points per bins in the reduction

#         bins is the discretization used to reduce the number of data points


#         Ninit_vertices is the max number of vertices in the mesh.
#             Final number of mesh ertices depend of the shape of the data and maximum number of vertices should be obtained for uniform data on a rectangle

#         alpha, strength of the mesh size dependance in local density of points. alpha=0 is uniform distribution of the mesh
#             negative alpha put more point in zone of low density and positive alpha put more points in zone of high density.
#             When alpha is 1.0, the local density of mesh point should be equal to the histogram of the data

#         h_min, ratio of smallest mesh edge over biggest mesh edge when using alpha != 0.

#     References
#     --------------

#     .. footbibliography::
#     """

#     from scipy.spatial import cKDTree
#     from scipy.integrate import cumulative_trapezoid
#     from .distmesh import distmesh, huniform, hdensity

#     dim = X.shape[1]

#     X, bbox, w, dx = reduce_data_size(X, bins, state_level)

#     if dim == 1:  # A adapter
#         from scipy.integrate import cumulative_trapezoid

#         # Use your existing 1D logic (optimized)
#         dhfun = w**alpha
#         h_vals = np.concatenate(([dhfun[0]], dhfun, [dhfun[-1]]))
#         x_vals = np.concatenate(([bbox[0]], X.ravel(), [bbox[1]]))
#         h_scaled = cumulative_trapezoid(h_vals, x_vals, initial=0)
#         h_scaled /= h_scaled[-1]
#         pts = np.interp(np.linspace(0, 1, Ninit_vertices), h_scaled, x_vals).reshape(-1, 1)
#         return pts, None
#     else:
#         tree = cKDTree(X)  # Un KDTree pour sélectionner efficacement un sous ensemble des points sur lequel faire la regression
#         bandwidth = np.linalg.norm(dx)

#         def density(x):
#             d, inds = tree.query(x, k=8)
#             Kw = (1 / np.sqrt(2 * np.pi)) * w[inds] * np.exp(-0.5 * (d / bandwidth) ** 2)
#             return 1e-2 - Kw.sum(axis=1)

#         def grad_log_density(x):
#             d, inds = tree.query(x, k=8)
#             Kw = (1 / np.sqrt(2 * np.pi)) * w[inds] * np.exp(-0.5 * (d / bandwidth) ** 2)
#             norm = Kw.sum(axis=1)
#             num = (Kw[..., None] * (x[:, None, :] - tree.data[inds])).sum(axis=1)
#             return -2 * np.divide(num, norm[:, None], out=np.zeros_like(x), where=norm[:, None] != 0)  # This is the local average of y

#         pfix = np.asarray(pfix).reshape(-1, dim)

#         rng = np.random.default_rng()
#         u = rng.uniform(0, 1, size=Ninit_vertices)  # Selection a random gaussian
#         cumsum_weight = np.cumsum(np.asarray(w))
#         sum_weight = cumsum_weight[-1]
#         i = np.searchsorted(cumsum_weight, u * sum_weight)

#         inits_points = np.atleast_2d(rng.normal(tree.data[i], bandwidth))
#         if dim == 2:
#             pts, tri = densmesh2d(inits_points, grad_log_density, density, pfix)
#         else:
#             h0 = (np.prod([np.abs(bbox[dim + n] - bbox[n]) for n in range(dim)]) / Ninit_vertices) ** 1 / dim
#             pts, tri = distmeshnd(dfunc, hdensity, h0, bbox, pfix)
#     if return_scaling:
#         return pts, tri, X, grad_log_density(X)
#     else:
#         return pts, tri


# if __name__ == "__main__":  # pragma: no cover
#     import matplotlib.pyplot as plt

#     points = np.random.rand(5000, 2)
#     vertices, tri = centroid_driven_mesh(points, 10, boundary_vertices=[[0, 0], [0, 1], [1, 0], [1, 1]])
#     plt.plot(points[:, 0], points[:, 1], "x")
#     plt.triplot(vertices[:, 0], vertices[:, 1], tri)
#     plt.plot(vertices[:, 0], vertices[:, 1], "o")
#     plt.show()
