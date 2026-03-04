"""
DistMesh Adapted from distmesh librairy from Per-Olof Persson and  Bradley Froehle
using Gemini to simplify the code

"""

import numpy as np
import scipy.spatial as spspatial
from scipy.spatial.distance import cdist
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# --- Distance Function Utilities ---


def huniform(p):
    """Implements the trivial uniform mesh size function h=1."""
    return np.ones(p.shape[0])


class hdensity:
    """
    Implement density based
    """

    def __init__(self, X, h0, bbox, min_scale=0.2, alpha=0.5):
        """
        Efficient sizing based on density rescaling.

        Parameters:
        -----------
        h0 : Background triangle size.
        min_scale : Finest triangle size relative to h0 (e.g., 0.2 * h0).
        alpha : Contrast control. 1.0 = Linear, <1.0 = Faster transition.
        """
        self.h0 = h0
        self.min_scale = min_scale

        # 1. Setup Grid Resolution tied to h0
        xmin, ymin, xmax, ymax = bbox
        # Padding by 2*h0 to handle boundary interpolation safely
        self.x_coords = np.arange(xmin - 2 * h0, xmax + 2 * h0, h0)
        self.y_coords = np.arange(ymin - 2 * h0, ymax + 2 * h0, h0)

        # 2. Compute Density (Histogram)
        hist, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[self.x_coords, self.y_coords])

        # 3. Smooth (Sigma=1.0 spreads density over ~3 cells/h0)
        smoothed_hist = gaussian_filter(hist, sigma=1.0)

        # 4. LINEAR RESCALING (No Clipping)
        # We map Density=0 to 1.0 and Density=Max to min_scale
        d_max = smoothed_hist.max()
        if d_max > 0:
            # Normalized density [0, 1]
            norm_d = (smoothed_hist / d_max) ** alpha
            # Linear map: h = 1.0 - (1.0 - min_scale) * norm_d
            sizing_grid = 1.0 - (1.0 - min_scale) * norm_d
        else:
            sizing_grid = np.ones_like(smoothed_hist)

        # 5. Fast Interpolator
        xc = (self.x_coords[:-1] + self.x_coords[1:]) / 2
        yc = (self.y_coords[:-1] + self.y_coords[1:]) / 2
        self.interp = RegularGridInterpolator((xc, yc), sizing_grid, method="linear", bounds_error=False, fill_value=1.0)

    def __call__(self, p):
        # This is the fh(p) function for DistMesh
        return self.interp(p)


def ddiff(d1, d2):
    return np.maximum(d1, -d2)


def dcircle(p, xc, yc, r):
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def dellipse(p, xc, yc, rx, ry):
    return np.sqrt(((p[:, 0] - xc) ** 2) / rx**2 + ((p[:, 1] - yc) ** 2) / ry**2) - 1


def drectangle(p, x1, x2, y1, y2):
    d1 = np.minimum(-y1 + p[:, 1], y2 - p[:, 1])
    d2 = np.minimum(d1, -x1 + p[:, 0])
    return -np.minimum(d2, x2 - p[:, 0])


def dintersect(d1, d2):
    return np.maximum(d1, d2)


def dunion(d1, d2):
    return np.minimum(d1, d2)


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


def fixmesh(p, t):
    """Remove duplicated/unused nodes and fix element orientation."""
    # Remove duplicates
    p, jx = np.unique(np.round(p, 12), axis=0, return_inverse=True)
    t = jx[t]
    # Fix orientation (2D only)
    if p.shape[1] == 2:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        d02 = p[t[:, 2]] - p[t[:, 0]]
        vol = (d01[:, 0] * d02[:, 1] - d01[:, 1] * d02[:, 0]) / 2
        flip = vol < 0
        t[flip, 0:2] = t[flip, 1::-1]
    return p, t


def distmesh(fd, fh, h0, bbox, pfix=None, max_iter=400, jshow=200, delta_t=0.2, ttol=0.1, Fscale=1.2):
    """
    Unified 2D/ND DistMesh implementation.
    """
    bbox = np.array(bbox).reshape(2, -1)
    dim = bbox.shape[1]
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0

    # 1. Generate initial grid
    if dim == 2:
        # Equilateral triangle grid
        x, y = np.mgrid[bbox[0, 0] : bbox[1, 0] : h0, bbox[0, 1] : bbox[1, 1] : h0 * np.sqrt(3) / 2]
        x[:, 1::2] += h0 / 2
        p = np.vstack((x.ravel(), y.ravel())).T
    else:
        # Cartesian grid for ND
        grid = [np.arange(bbox[0, i], bbox[1, i] + h0, h0) for i in range(dim)]
        p = np.array(np.meshgrid(*grid)).reshape(dim, -1).T

    # 2. Rejection method
    p = p[fd(p) < geps]
    r0 = 1 / fh(p) ** dim
    p = p[np.random.rand(len(p)) < r0 / r0.max()]

    if pfix is not None:
        pfix = np.atleast_2d(pfix)
        # Remove duplicate within pfix
        pfix = np.unique(pfix, axis=0, return_index=False, return_inverse=False)
        nfix = pfix.shape[0]
        # Combine and remove duplicates close to fixed points
        dists = cdist(p, pfix)
        keep = np.all(dists > h0 * 0.1, axis=1)  # keep points that have a distance > h0/10 from any pfix
        p = p[keep]
        p = np.vstack([pfix, p])
    else:
        nfix = 0

    pold = np.inf
    count = 0

    while count < max_iter:
        # 3. Retriangulation
        if np.max(np.linalg.norm(p - pold, axis=1)) > h0 * ttol:
            pold = p.copy()
            tri = spspatial.Delaunay(p)
            t = tri.simplices
            # Keep only interior triangles
            pmid = p[t].mean(axis=1)
            t = t[fd(pmid) < -geps]

            # Extract unique edges (bars)
            pairs = list(itertools.combinations(range(dim + 1), 2))
            edges = np.vstack([t[:, pair] for pair in pairs])
            edges.sort(axis=1)
            edges = np.unique(edges, axis=0)

        # 4. Forces
        barvec = p[edges[:, 0]] - p[edges[:, 1]]
        L = np.linalg.norm(barvec, axis=1)
        hbars = fh(p[edges].mean(axis=1))

        # Desired lengths L0
        L0 = hbars * Fscale * (np.sum(L**dim) / np.sum(hbars**dim)) ** (1 / dim)

        # Forces F (linear spring)
        F = np.maximum(L0 - L, 0)
        Fvec = (F / L)[:, None] * barvec

        # 5. Accumulate forces (replaces old 'dense' function)
        Ftot = np.zeros_like(p)
        np.add.at(Ftot, edges[:, 0], Fvec)
        np.add.at(Ftot, edges[:, 1], -Fvec)
        Ftot[:nfix] = 0  # Fixed points don't move

        p += delta_t * Ftot

        # 6. Project boundary points back
        d = fd(p)
        ix = d > 0
        if np.any(ix):
            dgrad = np.zeros((np.sum(ix), dim))
            for k in range(dim):
                step = np.zeros(dim)
                step[k] = deps
                dgrad[:, k] = (fd(p[ix] + step) - d[ix]) / deps
            p[ix] -= (d[ix, None] * dgrad) / np.sum(dgrad**2, axis=1)[:, None]

        # Termination criterion
        max_move = np.max(np.linalg.norm(delta_t * Ftot[d < -geps], axis=1))
        if max_move < 0.001 * h0:
            break
        if np.remainder(count, jshow) == 0:
            print("count = ", count, "N = ", p.shape[0], "displacement = ", max_move)
        count += 1

    return fixmesh(p, t)


def plot_mesh(p, t):
    if p.shape[1] != 2:
        print("Plotting only supported for 2D")
        return
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    poly = PolyCollection(p[t], edgecolors="black", facecolors="lightblue", linewidths=0.5)
    ax.add_collection(poly)
    ax.autoscale()
    plt.show()


# --- Example ---
if __name__ == "__main__":
    # Circle with a hole
    fd = lambda p: ddiff(dcircle(p, 0, 0, 1), dcircle(p, 0, 0, 0.4))
    fh = huniform
    p, t = distmesh(fd, fh, 0.1, [-1, -1, 1, 1])
    # plt.scatter(p[:, 0], p[:, 1])
    # plt.show()
    plot_mesh(p, t)
