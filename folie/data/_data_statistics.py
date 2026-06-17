from collections import namedtuple
from .._numpy import np
import scipy.stats
import scipy.optimize

DescribeResult = namedtuple("DescribeResult", ("nobs", "dim", "min", "max", "mean", "variance"))


def histogram(trajectories, bins, data_range=None):
    """
    Compute a multi-dimensional histogram of the trajectories.

    Parameters
    ----------
    trajectories : Trajectories or iterable of dict-like
        Each element must have an ``'x'`` key with shape (N, dim).
    bins : int or sequence of ints
        Number of bins per dimension (int) or bin edges per dimension (sequence).
    data_range : sequence of tuples, optional
        ``(min, max)`` per dimension. If None, uses ``trajectories.stats`` if available.

    Returns
    -------
    H : ndarray
        Accumulated histogram counts.
    edges : list of ndarray
        Bin edges for each dimension.
    OR (when hexbin=True):
    x_centers : ndarray
        X-coordinates of hexagon centers.
    y_centers : ndarray
        Y-coordinates of hexagon centers.
    C : ndarray
        Hexagon counts (1D array of non-empty bin counts).
    extent : tuple
        (xmin, xmax, ymin, ymax) of the hexbin extent.
    """
    # Determine range if not provided
    if data_range is None:
        stats = trajectories.stats
        data_range = [(stats.min[i], stats.max[i]) for i in range(stats.dim)]

    # Initialize histogram array
    if isinstance(bins, int):
        H = np.zeros((bins,) * len(data_range), dtype=float)
    else:
        H = np.zeros(tuple(len(b) - 1 for b in bins), dtype=float)

    # Accumulate histogram per trajectory
    for trj in trajectories:
        x = trj["x"]
        H += np.histogramdd(x, bins=bins, range=data_range)[0]

    # Compute bin edges
    if isinstance(bins, int):
        edges = [np.linspace(data_range[i][0], data_range[i][1], bins + 1) for i in range(len(data_range))]
    else:
        edges = list(bins)

    return H, edges


def hexbin(trajectories, gridsize):
    """
    Compute hexagonal binning incrementally over trajectories.

    Parameters
    ----------
    trajectories : Trajectories or iterable of dict-like
    gridsize : int
        Number of hexagons in the x-direction.
    range : sequence of tuples
        [(xmin, xmax), (ymin, ymax)].

    Returns
    -------
    x_centers : ndarray
        X-coordinates of hexagon centers.
    y_centers : ndarray
        Y-coordinates of hexagon centers.
    C : ndarray
        Hexagon counts (2D grid).
    extent : tuple
        (xmin, xmax, ymin, ymax).
    """

    stats = trajectories.stats
    data_range = [(stats.min[i], stats.max[i]) for i in range(stats.dim)]
    if len(data_range) != 2:
        raise ValueError("hexbin requires 2D data (data_range with 2 dimensions)")
    xmin, xmax = data_range[0]
    ymin, ymax = data_range[1]

    # Hexagon width
    hex_size = (xmax - xmin) / gridsize

    # Row spacing for hexagonal grid
    row_spacing = hex_size * np.sqrt(3) / 2

    # Number of rows needed
    num_rows = int(np.ceil((ymax - ymin) / row_spacing)) + 1

    # Initialize count array
    C = np.zeros((num_rows, gridsize), dtype=float)

    # Accumulate counts per trajectory
    for trj in trajectories:
        x = trj["x"][:, 0]
        y = trj["x"][:, 1]

        # Compute row indices
        rows = np.floor((y - ymin) / row_spacing).astype(int)
        rows = np.clip(rows, 0, num_rows - 1)

        # Compute column indices with odd-row offset
        offsets = (rows % 2) * hex_size / 2
        cols = np.floor((x - xmin - offsets) / hex_size).astype(int)
        cols = np.clip(cols, 0, gridsize - 1)

        # Flatten indices for bincount
        flat_indices = rows * gridsize + cols
        counts = np.bincount(flat_indices, minlength=num_rows * gridsize)

        # Reshape and add to C
        C += counts.reshape(num_rows, gridsize)

    extent = (xmin, xmax, ymin, ymax)

    # Return only non-empty bins (compatible with plt.hexbin x, y, C)
    mask = C > 0
    rows_idx, cols_idx = np.where(mask)

    y_centers = ymin + rows_idx * row_spacing
    offsets = (rows_idx % 2) * hex_size / 2
    x_centers = xmin + cols_idx * hex_size + offsets

    return x_centers, y_centers, C[mask], extent


def traj_stats(X):
    """
    Simply return the dimension of the data
    """
    if X.ndim == 2:
        nobs, dim = X.shape
    elif X.ndim == 1:
        nobs = X.shape[0]
        dim = 1
        X = X.reshape(-1, 1)
    else:
        nobs = X.shape[0]
        dim = X.shape[1]
    return DescribeResult(nobs, dim, np.asarray(X.min(0)), np.asarray(X.max(0)), np.asarray(X.mean(0)), np.asarray(X.var(0)))


def sum_stats(d1, d2):
    return DescribeResult(
        d1.nobs + d2.nobs,
        d1.dim,
        np.minimum(d1.min, d2.min),
        np.maximum(d1.max, d2.max),
        (d1.mean * d1.nobs + d2.mean * d2.nobs) / (d1.nobs + d2.nobs),
        ((d1.variance + d1.mean**2) * d1.nobs + (d2.variance + d2.mean**2) * d2.nobs) / (d1.nobs + d2.nobs) - ((d1.mean * d1.nobs + d2.mean * d2.nobs) / (d1.nobs + d2.nobs)) ** 2,
    )


def _beta_params_from_mean_var(mean, var, uniform_points, loc=0, scale=1, optimize=True):
    # Define the objective function to minimize
    def objective(x):
        a, b = x
        beta = scipy.stats.beta.ppf(uniform_points, a, b, loc=loc, scale=scale)
        return (beta.mean() - mean) ** 2 + (beta.var() - var) ** 2

    m = (mean - loc) / scale
    v = var / scale**2
    # Initial guess for parameters
    initial_guess = [m * (m * (1 - m) / v - 1.0), (1 - m) * (m * (1 - m) / v - 1.0)]
    if not optimize:
        return initial_guess

    # Constraints: a, b > 0
    constraints = [{"type": "ineq", "fun": lambda x: x[0]}, {"type": "ineq", "fun": lambda x: x[1]}]

    # Minimize the objective function
    result = scipy.optimize.minimize(objective, initial_guess, constraints=constraints)

    return result.x


def representative_array(stats, Npoints=75, optimize=False):
    """
    Build an array with the same statistics than stats with Npoints.
    This is an helper function to fit functions with a reduced number of points
    If optimize is True, then the parameters are ajusted to match the statistics otherwise this is an approximation
    """
    uniform = np.linspace(np.zeros_like(stats.min), np.ones_like(stats.max), Npoints)
    rep_array = np.empty_like(uniform)
    scale = stats.max - stats.min

    for d in range(stats.dim):
        a, b = _beta_params_from_mean_var(stats.mean[d], stats.variance[d], uniform[:, d], loc=stats.min[d], scale=scale[d], optimize=optimize)
        rep_array[:, d] = scipy.stats.beta.ppf(uniform[:, d], a, b, loc=stats.min[d], scale=scale[d])
    return rep_array


def mfpt_from_trajectories(data, state1_fn, state2_fn):
    """
    Compute Mean First Passage Time directly from trajectory data.

    States are defined by boolean mask functions that return True when
    the system is in that state.

    Parameters
    ----------
    data : folie.Trajectories
        Trajectory data object with dt attribute and trajectory arrays.
    state1_fn : callable
        Boolean function: position array -> boolean mask for state 1.
    state2_fn : callable
        Boolean function: position array -> boolean mask for state 2.

    Returns
    -------
    dict
        Dictionary with keys:
        - mfpt_1_to_2: mean first passage time from state 1 to state 2
        - mfpt_2_to_1: mean first passage time from state 2 to state 1
        - rates: transition rates (1/MFPT)
        - n_transitions_1_to_2: number of observed transitions
        - n_transitions_2_to_1: number of observed transitions

    Examples
    --------
    >>> def in_state_a(x):
    ...     return np.sum((x - center_a)**2, axis=1) < radius**2
    >>> def in_state_b(x):
    ...     return np.sum((x - center_b)**2, axis=1) < radius**2
    >>> result = mfpt_from_trajectories(data, in_state_a, in_state_b)
    >>> print(f"Rate A->B: {result['rates']['1_to_2']:.4f} /ps")
    """
    fpt_1_to_2 = []
    fpt_2_to_1 = []

    for traj in data:
        t = np.arange(len(traj["x"])) * data.dt
        x = traj["x"]

        # Boolean arrays for state membership
        s1 = state1_fn(x)
        s2 = state2_fn(x)

        # Label states: -1 for state 1, +1 for state 2, 0 for neither
        state_label = np.where(s1, -1, np.where(s2, 1, 0))

        # Find frames where system is inside a state
        if state_label.ndim == 0:
            continue
        non_zero_idx = np.where(state_label != 0)[0]
        if len(non_zero_idx) == 0:
            continue

        non_zero_states = state_label[non_zero_idx]

        # Find transitions (where the state changes between -1 and 1)
        changes = np.diff(non_zero_states) != 0
        change_idx = np.where(changes)[0]

        # Entry indices include first frame and transition points
        entry_indices = np.concatenate(([non_zero_idx[0]], non_zero_idx[change_idx + 1]))
        entered_states = state_label[entry_indices]

        fpts = np.diff(entry_indices) * data.dt

        # Route FPTs to correct list based on starting state
        starts_in = entered_states[:-1]

        fpt_1_to_2.extend(fpts[starts_in == -1])
        fpt_2_to_1.extend(fpts[starts_in == 1])

    mfpt_1_to_2 = np.mean(fpt_1_to_2) if fpt_1_to_2 else np.nan
    mfpt_2_to_1 = np.mean(fpt_2_to_1) if fpt_2_to_1 else np.nan

    rates = {}
    if mfpt_1_to_2 > 0:
        rates["1_to_2"] = 1.0 / mfpt_1_to_2
    if mfpt_2_to_1 > 0:
        rates["2_to_1"] = 1.0 / mfpt_2_to_1

    return {
        "mfpt_1_to_2": mfpt_1_to_2,
        "mfpt_2_to_1": mfpt_2_to_1,
        "rates": rates,
        "n_transitions_1_to_2": len(fpt_1_to_2),
        "n_transitions_2_to_1": len(fpt_2_to_1),
    }
