"""
Contain utilities function to find element corresponding to positions.
Allow for extrapolation of FEM by using the closest element when outside of the mesh

Adapted from scikit-fem
"""

import numpy as np
import skfem


def get_element_finder(mesh, mapping=None):
    """
    Get correct element finder depending of the type of the mesh
    """
    if isinstance(mesh, skfem.MeshLine):
        return element_finder_line(mesh, mapping)
    elif isinstance(mesh, skfem.MeshTri):
        return element_finder_tri(mesh, mapping)
    elif isinstance(mesh, skfem.MeshQuad):
        return element_finder_quad(mesh, mapping)
    elif isinstance(mesh, skfem.MeshTet):
        return element_finder_tet(mesh, mapping)
    elif isinstance(mesh, skfem.MeshHex):
        return element_finder_hex(mesh, mapping)
    elif isinstance(mesh, skfem.MeshWedge1):
        return element_finder_wedge(mesh, mapping)
    else:
        return mesh.element_finder(mapping)


def points_to_segment_dist(points, seg_start, seg_end):
    """Calculate the distance between points and segments."""
    seg_vectors = seg_end - seg_start
    points_vectors = points[:, :, None] - seg_start[:, None, :]

    seg_length_sq = np.sum(seg_vectors**2, axis=0)
    t = np.einsum("dij,dj->ij", points_vectors, seg_vectors) / seg_length_sq[None, :]
    t = np.clip(t, 0, 1)
    projections = seg_start[:, None, :] + t[None, :, :] * seg_vectors[:, None, :]
    distances_sq = np.sum((points[:, :, None] - projections) ** 2, axis=0)
    return distances_sq


def dist_tri_2d(x, y, tri):
    """
    Compute distance to edge
    tri is of shape (dim, 3, ntri)
    Shoud return array of shape (len(x), ntri)
    """
    # Juste pour tester on retourne la distance au centre
    points = np.array([x, y])
    distances = np.empty((x.shape[0], *tri.shape[1:]))
    for i in range(3):
        distances[:, i, :] = points_to_segment_dist(points, tri[:, i, :], tri[:, (i + 1) % 3, :])
    return np.min(distances, axis=1)

    # Using distances to center of triangle
    center_tri = np.mean(tri, axis=1)
    # print(center_tri.shape)
    dists = (x[:, None] - center_tri[0:1, :]) ** 2 + (y[:, None] - center_tri[1:2, :]) ** 2
    return dists


def points_to_facets_dist(points, triangles):
    """Calculate the distance between points and triangles in 3D in a vectorized manner."""

    v0 = triangles[:, 2] - triangles[:, 0]
    v1 = triangles[:, 1] - triangles[:, 0]
    normals = np.cross(v0, v1)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]

    points_vectors = points[:, None, :] - triangles[:, None, 0, :]
    dist_to_planes = np.einsum("ijk,ik->ij", points_vectors, normals)

    projections = points[:, None, :] - dist_to_planes[:, :, None] * normals[None, :, :]

    v2_proj = projections - triangles[:, None, 0, :]

    dot00 = np.einsum("ij,ij->i", v0, v0)
    dot01 = np.einsum("ij,ij->i", v0, v1)
    dot11 = np.einsum("ij,ij->i", v1, v1)
    dot02 = np.einsum("ijk,ik->ij", v2_proj, v0)
    dot12 = np.einsum("ijk,ik->ij", v2_proj, v1)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom[:, None]
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom[:, None]

    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    dist_to_planes_abs = np.abs(dist_to_planes)

    distances = np.full((points.shape[0], triangles.shape[0]), np.inf)

    distances[inside] = dist_to_planes_abs[inside]

    for i in range(3):
        seg_start = triangles[:, i]
        seg_end = triangles[:, (i + 1) % 3]
        dist_to_segments = points_to_segment_dist(points.T, seg_start.T, seg_end.T)
        distances = np.minimum(distances, dist_to_segments)

    min_distances = np.min(distances, axis=1)
    return min_distances


def dist_tet_3d(x, y, z, tet):
    """
    Compute distance to edge
    tri is of shape (dim, 4, ntet)
    Shoud return array of shape (len(x), ntet)
    """
    # Juste pour tester on retourne la distance au centre
    points = np.array([x, y, z])
    distances = np.empty((x.shape[0], *tet.shape[1:]))
    facets = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for i in range(4):
        distances[:, i, :] = points_to_facets_dist(points.T, tet[:, facets[i], :].T)
    return np.min(distances, axis=1)  # Distances to tetrahedral is shorst distance to facet

    # Using distances to center of tetrahedrals
    center_tri = np.mean(tet, axis=1)
    # print(center_tri.shape)
    dists = (x[:, None] - center_tri[0:1, :]) ** 2 + (y[:, None] - center_tri[1:2, :]) ** 2 + (z[:, None] - center_tri[2:3, :]) ** 2
    return dists


def element_finder_line(mesh, mapping=None):

    ix = np.argsort(mesh.p[0])
    maxt = mesh.t[np.argmax(mesh.p[0, mesh.t], 0), np.arange(mesh.t.shape[1])]

    def finder(x):
        xin = x.copy()  # bring endpoint inside for np.digitize
        xin[x == mesh.p[0, ix[-1]]] = mesh.p[0, ix[-2:]].mean()
        bins = np.digitize(xin, mesh.p[0, ix])
        bins[bins == 0] = 1
        bins[bins == len(mesh.p[0, ix])] = len(mesh.p[0, ix]) - 1
        elems = np.nonzero(ix[bins][:, None] == maxt)[1]
        if len(elems) < len(x):
            raise ValueError("Point is outside of the mesh.")
        return elems

    return finder


def element_finder_tri(mesh, mapping=None):

    if mapping is None:
        mapping = mesh._mapping()

    if not hasattr(mesh, "_cached_tree"):
        from scipy.spatial import cKDTree

        mesh._cached_tree = cKDTree(np.mean(mesh.p[:, mesh.t], axis=1).T)
    tree = mesh._cached_tree
    nelems = mesh.t.shape[1]

    def finder(x, y, _search_all=False):

        if not _search_all:
            ix = tree.query(np.array([x, y]).T, min(5, nelems))[1].flatten()
            _, ix_ind = np.unique(ix, return_index=True)
            ix = ix[np.sort(ix_ind)]
        else:
            ix = np.arange(nelems, dtype=np.int64)

        X = mapping.invF(np.array([x, y])[:, None], ix)
        eps = np.finfo(X.dtype).eps
        inside = (X[0] >= -eps) * (X[1] >= -eps) * (1 - X[0] - X[1] >= -eps)
        out_elems = ix[inside.argmax(axis=0)].flatten()

        if not inside.max(axis=0).all():
            if _search_all:
                outside = np.nonzero(~inside.max(axis=0))[0]
                x_out, y_out = x[outside], y[outside]
                # Not necessary to loop for all elements then
                ix_out = tree.query(np.array([x_out, y_out]).T, min(10, nelems))[1].flatten()
                _, ix_ind = np.unique(ix_out, return_index=True)
                ix_out = ix_out[np.sort(ix_ind)]

                dists = dist_tri_2d(x_out, y_out, mesh.p[:, mesh.t[:, ix_out]])
                out_elems[outside] = ix_out[np.argmin(dists, axis=1)]
                # raise ValueError("Point is outside of the mesh.")
            else:
                return finder(x, y, _search_all=True)

        return out_elems

    return finder


def element_finder_quad(mesh, mapping=None):
    """Transform to :class:`skfem.MeshTri` and return its finder."""
    tri_finder = mesh.to_meshtri().element_finder()

    def finder(*args):
        return tri_finder(*args) % mesh.t.shape[1]

    return finder


def element_finder_tet(mesh, mapping=None):

    if mapping is None:
        mapping = mesh._mapping()

    if not hasattr(mesh, "_cached_tree"):
        from scipy.spatial import cKDTree

        mesh._cached_tree = cKDTree(np.mean(mesh.p[:, mesh.t], axis=1).T)

    tree = mesh._cached_tree
    nelems = mesh.t.shape[1]

    def finder(x, y, z, _search_all=False):

        if not _search_all:
            ix = tree.query(np.array([x, y, z]).T, min(10, nelems))[1].flatten()
            _, ix_ind = np.unique(ix, return_index=True)
            ix = ix[np.sort(ix_ind)]
        else:
            ix = np.arange(nelems, dtype=np.int64)

        X = mapping.invF(np.array([x, y, z])[:, None], ix)
        eps = np.finfo(X.dtype).eps
        inside = (X[0] >= -eps) * (X[1] >= -eps) * (X[2] >= -eps) * (1 - X[0] - X[1] - X[2] >= -eps)
        out_elems = ix[inside.argmax(axis=0)].flatten()
        if not inside.max(axis=0).all():
            if _search_all:
                outside = np.nonzero(~inside.max(axis=0))[0]
                x_out, y_out, z_out = x[outside], y[outside], z[outside]
                # Not necessary to loop for all elements then
                ix_out = tree.query(np.array([x_out, y_out]).T, min(20, nelems))[1].flatten()
                _, ix_ind = np.unique(ix_out, return_index=True)
                ix_out = ix_out[np.sort(ix_ind)]

                dists = dist_tet_3d(x_out, y_out, z_out, mesh.p[:, mesh.t[:, ix_out]])
                out_elems[outside] = ix_out[np.argmin(dists, axis=1)]
                # raise ValueError("Point is outside of the mesh.")
            else:
                return finder(x, y, z, _search_all=True)

        return out_elems

    return finder


def element_finder_wedge(mesh, mapping=None):
    """Transform to :class:`skfem.MeshTet` and return its finder."""
    tet_finder = element_finder_tet(mesh.to_meshtet())

    def finder(*args):
        return tet_finder(*args) % mesh.t.shape[1]

    return finder


def element_finder_hex(mesh, mapping=None):
    """Transform to :class:`skfem.MeshTet` and return its finder."""
    tet_finder = element_finder_tet(mesh.to_meshtet())

    def finder(*args):
        return tet_finder(*args) % mesh.t.shape[1]

    return finder
