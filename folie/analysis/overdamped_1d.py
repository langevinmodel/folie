""" 
Set of analysis methods focused on 1D overdamped models
"""

from .._numpy import np
from scipy.integrate import cumulative_trapezoid


def free_energy_profile_1d(model, x):
    """
    From the force and diffusion construct the free energy profile
    """
    x = x.ravel()
    diff_prime_val = model.diffusion.grad_x(x.reshape(-1, 1)).ravel()
    force_val = model.force(x.reshape(-1, 1)).ravel()
    diff_val = model.diffusion(x.reshape(-1, 1)).ravel()

    diff_U = - force_val / diff_val +  diff_prime_val 
    
    pmf = cumulative_trapezoid(diff_U, x, initial=0.0)

    return pmf - np.min(pmf)


def mfpt_1d(model, x_start: float, x_abs: float, Npoints=500, cumulative=False):
    """
    Compute Mean first passage time from x0 to x1
    """
    min_x = np.min(self.knots_diff)

    x_range = np.linspace(min_x, x_abs, Npoints)

    prob_well = cumulative_trapezoid(np.exp(-self.free_energy_profile(x_range), x_range), initial=0.0)
    x_int = np.linspace(x_start, x_abs, Npoints)
    if cumulative:
        res = cumulative_trapezoid(np.exp(self.free_energy_profile(x_int)) * np.interp(x_int, x_range, prob_well) / self.diffusion(x_int, 0.0) ** 2, x_int, initial=0.0)
        return res
    else:
        return np.trapz(np.exp(self.free_energy_profile(x_int)) * np.interp(x_int, x_range, prob_well) / self.diffusion(x_int, 0.0) ** 2, x_int)
