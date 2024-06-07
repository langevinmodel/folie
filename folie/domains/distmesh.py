# Per-Olof Persson's code distmesh2D rewritten to Python and simplified
# File has been adapted from https://github.com/mar1akc/transition_path_theory_FEM_distmesh
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Delaunay
from scipy import sparse


def huniform(p):
    m, n = np.shape(p)
    return np.ones((m, 1))


def ddiff(d1, d2):
    return np.maximum(d1, -d2)


def dintersect(d1, d2):
    return np.maximum(d1, d2)


def dunion(d1, d2):
    return np.minimum(d1, d2)


def dcircle(p, xc, yc, r):
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def dellipse(p, xc, yc, rx, ry):
    return np.sqrt(((p[:, 0] - xc) ** 2) / rx**2 + ((p[:, 1] - yc) ** 2) / ry**2) - 1


def drectangle(p, x1, x2, y1, y2):
    d1 = np.minimum(-y1 + p[:, 1], y2 - p[:, 1])
    d2 = np.minimum(d1, -x1 + p[:, 0])
    return -np.minimum(d2, x2 - p[:, 0])


def dline(p, x1, y1, x2, y2):
    # signed distance from point p to line through (x1,y1) and  (x2,y2)
    # normal vector to the line
    nx = y1 - y2
    ny = x2 - x1
    nn = np.sqrt(nx * nx + ny * ny)
    # return (p-(x1,x2))*n/||n||
    return -((p[:, 0] - x1) * nx + (p[:, 1] - y1) * ny) / nn


def dtriangle(p, x1, y1, x2, y2, x3, y3):
    return np.maximum(dline(p, x1, y1, x2, y2), np.maximum(dline(p, x2, y2, x3, y3), dline(p, x3, y3, x1, y1)))


def triarea(pts, tri):
    # calculates areas of mesh triangles
    # p = [x_vec,y_vec]
    # tri = [ind0,ind1,ind2]
    d12 = pts[tri[:, 1], :] - pts[tri[:, 0], :]
    d13 = pts[tri[:, 2], :] - pts[tri[:, 0], :]
    A = d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]
    return A


def fixmesh(pts, tri):
    TOL = 1.0e-10
    # remove repeated nodes
    pts, idx = np.unique(pts, axis=0, return_inverse=True)
    tri = np.reshape(idx[tri], np.shape(tri))

    # compute areas of mesh triangles
    A = triarea(pts, tri)
    idx_tri_reorder = np.argwhere(A < 0)
    Nidx = np.size(idx_tri_reorder)
    idx_tri_reorder = np.reshape(idx_tri_reorder, (Nidx,))
    if np.any(idx_tri_reorder):
        # reorder triangles with negative area
        tmp = tri[idx_tri_reorder, 0]
        tri[idx_tri_reorder, 0] = tri[idx_tri_reorder, 1]
        tri[idx_tri_reorder, 1] = tmp
    # remove triangles with too small area
    idx_keep = np.argwhere(np.absolute(A) > TOL * np.linalg.norm(A, np.inf))
    Nidx = np.size(idx_keep)
    idx_keep = np.reshape(idx_keep, (Nidx,))
    tri = tri[idx_keep, :]
    # remove unused nodes
    Ntri, m = np.shape(tri)
    t_col = np.reshape(tri, (Ntri * m,))
    idx, iidx = np.unique(t_col, return_inverse=True)
    pts = pts[idx, :]
    tri = np.reshape(iidx, (Ntri, m))
    return pts, tri


def distmesh2D(fd, fh, h0, bbox, pfix, **kwargs):
    """
    Generate 2D Mesh from a distance function.

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
        x1, x2, y1, y2 = 0.0, 1.0, -1.0, 1.0
        def dfunc(p):
            d0 = drectangle(p, x1, x2, y1, y2)
            dA = dellipse(p, xa, ya, rx, ry)
            dB = dellipse(p, xb, yb, rx, ry)
            d = ddiff(d0, dunion(dA, dB))
            return d

        # h0 is the desired scaling parameter for the mesh
        h0 = 0.04
        Nfix = Na + Nb + Nouter
        # bbox = [xmin,xmax,ymin,ymax]
        bbox = [xmin, xmax, ymin, ymax]
        pts, tri = distmesh2D(dfunc, huniform, h0, bbox, [])
        mesh = meshio.Mesh(pts, [("triangle", tri)])
        print(mesh)
        mesh.write("mesh.vtk")
    """
    dim = 2
    # parameters
    dptol = 0.001
    ttol = 0.1
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = math.sqrt(np.finfo(float).eps) * h0
    MAXcount = kwargs.get("MAXcount", 5000)
    densityctrlfreq = 30
    jshow = 200  # display progress every jshow iterations

    # define the initial set of points by
    # making a mesh of equilateral triangles with side h0 and
    # adding fixed points

    ax = np.arange(bbox[0], bbox[1], h0)
    ay = np.arange(bbox[2], bbox[3], h0 * math.sqrt(3) * 0.5)
    x, y = np.meshgrid(ax, ay)
    nx, ny = np.shape(x)
    nxy = nx * ny
    x[1:nx:2, :] = x[1:nx:2, :] + h0 * 0.5  # Shift odd rows
    x_vec = np.reshape(x, (nxy, 1))
    y_vec = np.reshape(y, (nxy, 1))
    pts = np.concatenate((x_vec, y_vec), axis=1)  # List of node coordinates
    # remove points outside the region
    jremove = np.argwhere(fd(pts) > geps)
    Nj = np.size(jremove)
    jremove = np.reshape(jremove, (Nj,))
    pts = np.delete(pts, jremove, 0)
    if np.any(pfix):  # if pfix is nonempty, i.e., there are fixed points
        pfix = np.unique(pfix, axis=0)  # extract unique rows in pfix
        nfix, d = np.shape(pfix)
        pts = np.concatenate((pfix, pts), axis=0)  # prepend fixed points
    else:
        nfix = 0

    Npts = np.size(pts, 0)  # the number of points

    count = 0
    displacement = math.inf
    pts_old = math.inf

    while displacement > dptol and count < MAXcount:
        count = count + 1
        if max(np.sqrt(np.sum((pts - pts_old) ** 2, axis=1)) / h0) > ttol:
            pts_old = pts
            tri = Delaunay(pts).simplices
            pts_ctr = (pts[tri[:, 0], :] + pts[tri[:, 1], :] + pts[tri[:, 2], :]) / 3  # centroids of triangles
            tri = tri[fd(pts_ctr) < -geps, :]  # keep only interior triangles
            Ntri = np.size(tri, axis=0)
            bars = np.concatenate((tri[:, [0, 1]], tri[:, [0, 2]]), axis=0)
            bars = np.concatenate((bars, tri[:, [1, 2]]), axis=0)
            bars = np.unique(np.sort(bars, axis=1), axis=0)
            Nbars, d = np.shape(bars)

        # move mesh points based on bar lengths L and forces F
        barvec = pts[bars[:, 0], :] - pts[bars[:, 1], :]  # List of bar vectors
        L = np.sqrt(np.sum(barvec**2, axis=1))  # L = Bar lengths
        L = np.reshape(L, (Nbars, 1))
        hbars = fh((pts[bars[:, 0], :] + pts[bars[:, 1], :]) / 2)
        L0 = hbars * Fscale * (sum(L**dim) / np.sum(hbars**dim)) ** ((1.0).dim)  # L0 = Desired lengths
        L0 = np.reshape(L0, (Nbars, 1))

        # density control: remove points if they are too close
        if np.remainder(count, densityctrlfreq) == 0 and np.any(L0 > 2 * L):
            jremove = np.argwhere(L0 > 2 * L)
            Nj = np.size(jremove)
            jremove = np.reshape(jremove, (Nj,))
            jremove = np.unique(np.reshape(bars[jremove, :], (Nj * 2,)))
            jremove = np.setdiff1d(jremove, np.arange(nfix))
            pts = np.delete(pts, jremove, axis=0)
            Npts, d = np.shape(pts)  # the number of points
            pts_old = math.inf
            continue

        F = np.maximum(L0 - L, np.zeros_like(L0))
        Fvec = np.matmul(F / L, np.ones((1, dim))) * barvec  # Bar forces (x,y components)
        Ftot = np.zeros_like(pts)
        for ndim in range(dim):
            I = bars[:, 0]
            J = ndim * np.ones_like(I, dtype=int)
            V = Fvec[:, ndim]
            Ftot += sparse.coo_matrix((V, (I, J)), shape=(Npts, dim)).toarray()
            I = bars[:, 1]
            J = ndim * np.ones_like(I, dtype=int)
            V = Fvec[:, ndim]
            Ftot -= sparse.coo_matrix((V, (I, J)), shape=(Npts, dim)).toarray()

        Ftot[0:nfix, :] = 0  # force = 0 at fixed points
        pts = pts + deltat * Ftot  # Update node positions

        # Bring outside points back to the boundary
        d = fd(pts)
        idx = np.argwhere(d > 0)  # find points outside the domain
        Nidx = np.size(idx)
        idx = np.reshape(idx, (Nidx,))

        dx = np.zeros_like(pts[idx, :])
        dx[:, 0] = deps
        dgradx = (fd(pts[idx, :] + dx) - d[idx]) / deps
        dy = np.zeros_like(pts[idx, :])
        dy[:, 1] = deps
        dgrady = (fd(pts[idx, :] + dy) - d[idx]) / deps
        dgrad2 = dgradx**2 + dgrady**2
        pts[idx, :] = pts[idx, :] - np.concatenate((np.reshape(d[idx] * dgradx / dgrad2, (Nidx, 1)), np.reshape(d[idx] * dgrady / dgrad2, (Nidx, 1))), axis=1)

        # termination criterion
        idx = np.argwhere(d < -geps)  # find interior nodes
        Nidx = np.size(idx)
        idx = np.reshape(idx, (Nidx,))
        if Nidx > 0:
            displacement = np.amax(np.sqrt(np.sum(deltat * Ftot[idx, :] ** 2, axis=1)) / h0)  # mamimal displacement, scaled
        else:
            displacement = 0.0
        if np.remainder(count, jshow) == 0:
            print("count = ", count, "displacement = ", displacement)

    pts, tri = fixmesh(pts, tri)
    plt.triplot(pts[:, 0], pts[:, 1], tri, linewidth=0.1)
    axes = plt.gca()
    axes.set_aspect(1)
    return pts, tri
