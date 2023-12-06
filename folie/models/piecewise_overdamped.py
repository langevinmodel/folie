from .overdamped import ModelOverdamped
import numpy as np
import numba as nb


@nb.njit
def linear_interpolation_with_gradient(idx, h, knots, fp):
    n_knots = knots.shape[0]
    f0, f1 = fp[idx - 1], fp[idx]
    # Second parameter set is in second half of array
    g0, g1 = fp[idx - 1 + n_knots], fp[idx + n_knots]

    hm = 1 - h
    val_f = hm * f0 + h * f1
    val_g = hm * g0 + h * g1

    # Set gradient elements one by one
    grad = np.zeros((knots.shape[0], idx.shape[0]))
    for i, ik in enumerate(idx):
        grad[ik - 1, i] = hm[i]
        grad[ik, i] = h[i]
    return val_f, val_g, grad


class OverdampedFreeEnergy(ModelOverdamped):
    """
    TODO: A class that implement a overdamped model with a given free energy
    """

    def __init__(self, knots, beta, **kwargs):
        super().__init__()
        self.knots = knots.ravel()
        self.beta = beta
        self._n_coeffs_force = len(self.knots)
        self.coefficients = np.concatenate((np.zeros(self._n_coeffs_force), np.ones(self._n_coeffs_force)))

    def preprocess_traj(self, x, use_midpoint=False, **kwargs):
        """Preprocess colvar trajectory with a given grid for faster model optimization

        Args:
            q (list of ndarray): trajectories of the CV.
            knots (ndarray): CV values forming the knots of the piecewise-linear approximation of logD and gradF.

        Returns:
            traj (numba types list): list of tuples (bin indices, bin positions, displacements)
        """

        # TODO: enable subsampling by *averaging* biasing force in interval
        # Then run inputting higher-res trajectories
        trj = x.ravel()
        deltaq = trj[1:] - trj[:-1]

        if use_midpoint:
            # Use mid point of each interval
            # Implies a "leapfrog-style" integrator that is not really used for overdamped LE
            ref_q = 0.5 * (trj[:-1] + trj[1:])
        else:
            # Truncate last traj point to match deltaq array
            ref_q = trj[:-1]

        # bin index on possibly irregular grid
        idx = np.searchsorted(self.knots, ref_q)
        assert (idx > 0).all() and (idx < len(self.knots)).all(), "Out-of-bounds point(s) in trajectory\n"
        # # Other option: fold back out-of-bounds points - introduces biases
        # idx = np.where(idx == 0, 1, idx)
        # idx = np.where(idx == len(knots), len(knots) - 1, idx)

        q0, q1 = self.knots[idx - 1], self.knots[idx]
        # fractional position within the bin
        h = (trj[:-1] - q0) / (q1 - q0)

        # Numba prefers typed lists
        return (idx, h, deltaq)

    def force(self, x, t: float = 0.0):
        idx, h, _ = self.preprocess_traj(x)
        G, logD, _ = linear_interpolation_with_gradient(idx, h, self.knots, self._coefficients)
        return -self.beta * np.exp(logD) * G

    def diffusion(self, x, t: float = 0.0):
        idx, h, _ = self.preprocess_traj(x)
        G, logD, _ = linear_interpolation_with_gradient(idx, h, self.knots, self._coefficients)
        return 2.0 * np.exp(logD)

    def force_t(self, x, t: float = 0.0):
        return 0.0

    def force_x(self, x, t: float = 0.0):
        idx, h, _ = self.preprocess_traj(x)
        G, logD, dXdk = linear_interpolation_with_gradient(idx, h, self.knots, self._coefficients)
        return np.dot(self._coefficients[: self._n_coeffs_force], self.basis.derivative(x))

    def force_xx(self, x, t: float = 0.0):
        return 0.0

    def diffusion_t(self, x, t: float = 0.0):
        return 0.0

    def diffusion_x(self, x, t: float = 0.0):
        idx, h, _ = self.preprocess_traj(x)
        G, logD, dXdk = linear_interpolation_with_gradient(idx, h, self.knots, self._coefficients)
        return np.dot(self._coefficients[self._n_coeffs_force :], self.basis.derivative(x))

    def diffusion_xx(self, x, t: float = 0.0):
        return 0.0

    def is_linear(self):
        return True
