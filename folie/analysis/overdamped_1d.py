"""
Set of analysis methods focused on 1D overdamped models
"""

from .._numpy import np
from scipy.integrate import cumulative_trapezoid, solve_ivp


def free_energy_profile_1d(model, x):
    r"""
    From the drift F(x) and diffusion D(x) construct the free energy profile V(x) using the formula

    .. math::
        F(x) = -D(x) \nabla V(x) + \mathrm{div} D(x)

    """
    x = x.ravel()

    def grad_V(x, _):
        x = np.asarray(x).reshape(-1, 1)
        diff_prime_val = model.diffusion.grad_x(x).ravel()
        drift_val = model.drift(x).ravel()
        diff_val = model.diffusion(x).ravel()

        return (-drift_val + diff_prime_val) / diff_val

    sol = solve_ivp(grad_V, [x.min() - 1e-10, x.max() + 1e10], np.array([0.0]), t_eval=x)  # Add some epsilon to range to ensure inclusion of x

    V = sol.y.ravel()

    # V = cumulative_trapezoid(grad_V(x), x, initial=0.0)

    return V - np.min(V)


def mfpt_1d(model, x_end: float, x_range, Npoints=500, x_start=None):
    r"""
    Compute the mean first passage time from any point x within x_range to x_end, or from x_start to x_end if x_start is defined.

    It use numerical integration of the following formula for point from x_range[0] to x_end :footcite:p:`Jungblut2016`

    .. math::
        MFPT(x,x_{end}) = \int_x^{x_{end}} \mathrm{d}y \frac{e^{\beta V(y)}}{D(y)} \int_{x\_range[0]}^y \mathrm{d} z e^{-\beta V(y)}

    and for point from x_end to x_range[1]

    .. math::

        MFPT(x,x_{end}) = \int^x_{x_{end}} \mathrm{d}y \frac{e^{\beta V(y)}}{D(y)} \int^{x\_range[1]}_y \mathrm{d} z e^{-\beta V(y)}

    Parameters
    ------------

        model:
            A fitted overdamped model

        x_end: float
            The point to reach

        x_start: float, default to None
            If this is not None it return the MFPT from x_start to x_end, otherwise it return the mean first passage from any point within x_range to x_end

        x_range:
            A range of integration, It should be big enough to be able to compute the normalisation factor of the steady state probability.

        Npoints: int, default=500
            Number of point to use for

    References
    --------------

    .. footbibliography::

    """

    if x_start is not None:
        if x_start < x_end:
            int_range = np.linspace(x_range[0], x_end, Npoints)
            prob_well = cumulative_trapezoid(np.exp(-free_energy_profile_1d(model, int_range)), int_range, initial=0.0)
        else:
            int_range = np.linspace(x_end, x_range[1], Npoints)
            prob_well = cumulative_trapezoid(np.exp(-free_energy_profile_1d(model, int_range)), int_range, initial=0.0)
            prob_well -= prob_well[-1]

        x_int = np.linspace(x_start, x_end, Npoints)
        return np.trapz(np.exp(free_energy_profile_1d(model, x_int)) * np.interp(x_int, int_range, prob_well) / model.diffusion(x_int.reshape(-1, 1)).ravel(), x_int)

    else:
        # Compute lower part
        int_range = np.linspace(x_range[0], x_end, Npoints)
        prob_well = cumulative_trapezoid(np.exp(-free_energy_profile_1d(model, int_range)), int_range, initial=0.0)

        x_int_lower = np.linspace(x_end, x_range[0], Npoints)
        res_lower = -1 * cumulative_trapezoid(np.exp(free_energy_profile_1d(model, x_int_lower)) * np.interp(x_int_lower, int_range, prob_well) / model.diffusion(x_int_lower.reshape(-1, 1)).ravel(), x_int_lower, initial=0.0)

        int_range = np.linspace(x_end, x_range[1], Npoints)
        prob_well = -1 * cumulative_trapezoid(np.exp(-free_energy_profile_1d(model, int_range)), int_range, initial=0.0)
        prob_well -= prob_well[-1]
        # return (int_range, prob_well)
        x_int_upper = np.linspace(x_end, x_range[1], Npoints)
        res_upper = cumulative_trapezoid(np.exp(free_energy_profile_1d(model, x_int_upper)) * np.interp(x_int_upper, int_range, prob_well) / model.diffusion(x_int_upper.reshape(-1, 1)).ravel(), x_int_upper, initial=0.0)
        return np.hstack((x_int_lower[::-1], x_int_upper)), np.hstack((res_lower[::-1], res_upper))
