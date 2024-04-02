import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay


def non_uniform_line(x_start, x_end, num_elements, ratio):
    """
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
    """
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


if __name__ == "__main__":  # pragma: no cover
    import matplotlib.pyplot as plt

    points = np.random.rand(5000, 2)
    vertices, tri = centroid_driven_mesh(points, 10, boundary_vertices=[[0, 0], [0, 1], [1, 0], [1, 1]])
    plt.plot(points[:, 0], points[:, 1], "x")
    plt.triplot(vertices[:, 0], vertices[:, 1], tri)
    plt.plot(vertices[:, 0], vertices[:, 1], "o")
    plt.show()
