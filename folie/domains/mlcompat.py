"""
MATLAB compatibility methods

dense          : Similar to full(sparse(I, J, S, ...))
interp2_linear : Similar to interp2(..., 'linear')
interp3_linear : Similar to interpn(..., 'linear') for dim=3
unique_rows    : Similar to unique(..., 'rows')
setdiff_rows   : Similar to setdiff(..., 'rows')
"""

import numpy as np
import scipy.sparse as spsparse
import scipy.interpolate as spinterp

import collections
import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib.collections import PathCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path


# -----------------------------------------------------------------------------
# 3D Plotting
# -----------------------------------------------------------------------------


def _mkfaces(t):
    """All exterior faces, as a set."""
    return set(tuple(sorted(f)) for f in dmutils.boundedgesnd(t))


def _trimesh(ax, t, x, y, z, **kwargs):
    """Display a 3D triangular mesh.

    Ignores ax._hold.
    """
    from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d

    patches = []
    code = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    for f in t:
        bdry = np.take(f, range(4), mode="wrap")
        pp = PathPatch(Path(np.column_stack((x[bdry], y[bdry])), code), **kwargs)
        ax.add_patch(pp)
        pathpatch_2d_to_3d(pp, z[bdry])
        patches.append(pp)
    return patches


def axes_simpplot3d(ax, p, t, pmask=None, **kwargs):
    """Plot a surface or volume triangulation.

    Parameters
    ----------
    p : array, shape (np, 3)
    t : array, shape (nt, 3) or (nt, 4)
    pmask : callable or bool array of shape (np,)

    Additional keyword arguments
    ----------------------------
    facecolor : facecolor
    ifacecolor : facecolor for faces exposed by pmask
    """
    if not ax._hold:
        ax.cla()
    had_data = ax.has_data()

    facecolor = kwargs.pop("facecolor", (0.8, 0.9, 1.0))
    ifacecolor = kwargs.pop("ifacecolor", (0.9, 0.8, 1.0))
    xs, ys, zs = p.T

    ret = cbook.silent_list("mpl_toolkits.mplot3d.art3d.PathPatch3D")

    if t.shape[1] == 4:
        tri1 = _mkfaces(t)

        if pmask is not None:
            if isinstance(pmask, collections.Callable):
                pmask = pmask(p)
            t = t[pmask[t].any(1)]
            tri2 = _mkfaces(t)
            tri1 = tri1.intersection(tri2)
            tri2 = tri2.difference(tri1)
            c = _trimesh(ax, tri2, xs, ys, zs, facecolor=ifacecolor)
            ret.extend(c)
    else:
        tri1 = t
        if pmask is not None:
            if isinstance(pmask, collections.Callable):
                pmask = pmask(p)
            tri1 = t[pmask[t].any(1)]

    c = _trimesh(ax, tri1, xs, ys, zs, facecolor=facecolor)
    ret.extend(c)
    ax.auto_scale_xyz(xs, ys, zs, had_data)

    return ret


class SimplexCollection(PathCollection):
    """A collection of triangles."""

    def __init__(self, simplices=None, **kwargs):
        kwargs.setdefault("linewidths", 0.5)
        kwargs.setdefault("edgecolors", "k")
        kwargs.setdefault("facecolors", (0.8, 0.9, 1.0))
        PathCollection.__init__(self, [], **kwargs)
        if simplices is not None:
            self.set_simplices(simplices)

    def set_simplices(self, simplices):
        """Usage: set_simplices((p, t))"""
        p, t = simplices
        code = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        self.set_paths([Path(edge, code) for edge in p[t[:, [0, 1, 2, 0]]]])


def simpvol(p, t):
    """Signed volumes of the simplex elements in the mesh."""
    dim = p.shape[1]
    if dim == 1:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        return d01
    elif dim == 2:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        d02 = p[t[:, 2]] - p[t[:, 0]]
        return (d01[:, 0] * d02[:, 1] - d01[:, 1] * d02[:, 0]) / 2
    else:
        raise NotImplementedError


def fixmesh(p, t, ptol=2e-13):
    """Remove duplicated/unused nodes and fix element orientation.

    Parameters
    ----------
    p : array, shape (np, dim)
    t : array, shape (nt, nf)

    Usage
    -----
    p, t = fixmesh(p, t, ptol)
    """
    snap = (p.max(0) - p.min(0)).max() * ptol
    _, ix, jx = unique_rows(np.round(p / snap) * snap, True, True)

    p = p[ix]
    t = jx[t]

    flip = simpvol(p, t) < 0
    t[flip, :2] = t[flip, 1::-1]

    return p, t


def dense(I, J, S, shape=None, dtype=None):
    """
    Similar to MATLAB's SPARSE(I, J, S, ...), but instead returning a
    dense array.

    Usage
    -----
    >>> shape = (m, n)
    >>> A = dense(I, J, S, shape, dtype)
    """

    # Advanced usage: allow J and S to be scalars.
    if np.isscalar(J):
        x = J
        J = np.empty(I.shape, dtype=int)
        J.fill(x)
    if np.isscalar(S):
        x = S
        S = np.empty(I.shape)
        S.fill(x)

    # Turn these into 1-d arrays for processing.
    S = S.flat
    I = I.flat
    J = J.flat
    return spsparse.coo_matrix((S, (I, J)), shape, dtype).toarray()


def interp2_linear(x, y, z, xi, yi):
    """
    Similar to interp2(..., '*linear') in MATLAB.

    Uses x,y,z to construct f, a linear function satisfying
        z[i, j] = f(x[i], y[j])

    Then returns zi, and array found by evaluating f:
        zi[i] = f(xi[i], yi[i])

    Parameters
    ----------
    x, y : array, ndim=1
    z : array, shape (x.size, y.size)
    xi, yi : array, shape (n,)

    Returns
    -------
    zi : array, shape (n,)
    """
    return spinterp.RectBivariateSpline(x, y, z, kx=1, ky=1).ev(xi, yi)


def interp3_linear(x, y, z, w, xi, yi, zi):
    """Similar to interpn(..., '*linear') in MATLAB for dim=3"""
    p = np.vstack((x.flat, y.flat, z.flat)).T
    v = w.flaten()
    f = spinterp.LinearNDInterpolator(p, v)

    pi = np.vstack((xi.flat, yi.flat, zi.flat)).T
    return f(pi)


def setdiff_rows(A, B, return_index=False):
    """
    Similar to MATLAB's setdiff(A, B, 'rows'), this returns C, I
    where C are the row of A that are not in B and I satisfies
    C = A[I,:].

    Returns I if return_index is True.
    """
    A = np.require(A, requirements="C")
    B = np.require(B, requirements="C")

    assert A.ndim == 2, "array must be 2-dim'l"
    assert B.ndim == 2, "array must be 2-dim'l"
    assert A.shape[1] == B.shape[1], "arrays must have the same number of columns"
    assert A.dtype == B.dtype, "arrays must have the same data type"

    # NumPy provides setdiff1d, which operates only on one dimensional
    # arrays. To make the array one-dimensional, we interpret each row
    # as being a string of characters of the appropriate length.
    orig_dtype = A.dtype
    ncolumns = A.shape[1]
    dtype = np.dtype((np.character, orig_dtype.itemsize * ncolumns))
    C = np.setdiff1d(A.view(dtype), B.view(dtype)).view(A.dtype).reshape((-1, ncolumns), order="C")
    if return_index:
        raise NotImplementedError
    else:
        return C


def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]

    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements="C")
    assert A.ndim == 2, "array must be 2-dim'l"

    orig_dtype = A.dtype
    ncolumns = A.shape[1]
    dtype = np.dtype((np.character, orig_dtype.itemsize * ncolumns))
    B, I, J = np.unique(A.view(dtype), return_index=True, return_inverse=True)

    B = B.view(orig_dtype).reshape((-1, ncolumns), order="C")

    # There must be a better way to do this:
    if return_index:
        if return_inverse:
            return B, I, J
        else:
            return B, I
    else:
        if return_inverse:
            return B, J
        else:
            return B
