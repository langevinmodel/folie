"""
Somes utilities function
"""
import numpy as np


def filter_kalman(mutm, Sigtm, Xt, mutilde_tm, expAh, SST, dim_x, dim_h):
    """
    Compute the foward step using Kalman filter, predict and update step

    Parameters
    ----------
    mutm, Sigtm: Values of the foward distribution at t-1
    Xt, mutilde_tm: Values of the trajectories at T and t-1
    expAh, SST: Coefficients parameters["expA"][:, dim_x:] (dim_x+dim_h, dim_h) and SS^T (dim_x+dim_h, dim_x+dim_h)
    dim_x,dim_h: Dimension of visibles and hidden variables
    """
    # Predict step marginalization Normal Gaussian
    mutemp = mutilde_tm + np.matmul(expAh, mutm)
    Sigtemp = SST + np.matmul(expAh, np.matmul(Sigtm, expAh.T))
    # Update step conditionnal Normal Gaussian
    invSYY = np.linalg.inv(Sigtemp[:dim_x, :dim_x])
    marg_mu = mutemp[dim_x:] + np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, Xt - mutemp[:dim_x]))
    marg_sig = Sigtemp[dim_x:, dim_x:] - np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, Sigtemp[dim_x:, :dim_x].T))
    marg_sig = 0.5 * marg_sig + 0.5 * marg_sig.T  # Enforce symmetry
    R = expAh[dim_x:, :] - np.matmul(Sigtemp[dim_x:, :dim_x], np.matmul(invSYY, expAh[:dim_x, :]))
    # Pair probability distibution Z_t,Z_{t-1}
    mu_pair = np.hstack((marg_mu, mutm))
    Sig_pair = np.zeros((2 * dim_h, 2 * dim_h))
    Sig_pair[:dim_h, :dim_h] = marg_sig
    Sig_pair[dim_h:, :dim_h] = np.matmul(R, Sigtm)
    Sig_pair[:dim_h, dim_h:] = Sig_pair[dim_h:, :dim_h].T
    Sig_pair[dim_h:, dim_h:] = Sigtm

    return marg_mu, marg_sig, mu_pair, Sig_pair


def smoothing_rauch(muft, Sigft, muStp, SigStp, Xtplus, mutilde_t, expAh, SST, dim_x, dim_h):
    """
    Compute the backward step using Kalman smoother
    """

    invTemp = np.linalg.inv(SST + np.matmul(expAh, np.matmul(Sigft, expAh.T)))
    R = np.matmul(np.matmul(Sigft, expAh.T), invTemp)

    mu_dym_rev = muft + np.matmul(R[:, :dim_x], Xtplus) - np.matmul(R, np.matmul(expAh, muft) + mutilde_t)
    Sig_dym_rev = Sigft - np.matmul(np.matmul(R, expAh), Sigft)

    marg_mu = mu_dym_rev + np.matmul(R[:, dim_x:], muStp)
    marg_sig = np.matmul(R[:, dim_x:], np.matmul(SigStp, R[:, dim_x:].T)) + Sig_dym_rev

    # Pair probability distibution Z_{t+1},Z_{t}
    mu_pair = np.hstack((muStp, marg_mu))
    Sig_pair = np.zeros((2 * dim_h, 2 * dim_h))
    Sig_pair[:dim_h, :dim_h] = SigStp
    Sig_pair[dim_h:, :dim_h] = np.matmul(R[:, dim_x:], SigStp)
    Sig_pair[:dim_h, dim_h:] = Sig_pair[dim_h:, :dim_h].T
    Sig_pair[dim_h:, dim_h:] = marg_sig

    return marg_mu, marg_sig, mu_pair, Sig_pair


def filtersmoother(Xtplus, mutilde, R, diffusion_coeffs, mu0, sig0):
    """
    Apply Kalman filter and Rauch smoother. Fallback for the fortran implementation
    """
    # Initialize, we are going to use a numpy array for storing intermediate values and put the resulting µh and \Sigma_h into the xarray only at the end
    lenTraj = Xtplus.shape[0]
    dim_x = Xtplus.shape[1]
    dim_h = mu0.shape[0]

    muf = np.zeros((lenTraj, dim_h))
    Sigf = np.zeros((lenTraj, dim_h, dim_h))
    mus = np.zeros((lenTraj, dim_h))
    Sigs = np.zeros((lenTraj, dim_h, dim_h))
    # To store the pair probability distibution
    muh = np.zeros((lenTraj, 2 * dim_h))
    Sigh = np.zeros((lenTraj, 2 * dim_h, 2 * dim_h))

    # Forward Proba
    muf[0, :] = mu0
    Sigf[0, :, :] = sig0
    # Iterate and compute possible value for h at the same point
    for i in range(1, lenTraj):
        # try:
        muf[i, :], Sigf[i, :, :], muh[i - 1, :], Sigh[i - 1, :, :] = filter_kalman(muf[i - 1, :], Sigf[i - 1, :, :], Xtplus[i - 1], mutilde[i - 1], R, diffusion_coeffs, dim_x, dim_h)
        # except np.linalg.LinAlgError:
        #     print(i, muf[i - 1, :], Sigf[i - 1, :, :], Xtplus[i - 1], mutilde[i - 1], self.friction_coeffs[:, self.dim_x :], self.diffusion_coeffs)
    # The last step comes only from the forward recursion
    Sigs[-1, :, :] = Sigf[-1, :, :]
    mus[-1, :] = muf[-1, :]
    # Backward proba
    for i in range(lenTraj - 2, -1, -1):  # From T-1 to 0
        # try:
        mus[i, :], Sigs[i, :, :], muh[i, :], Sigh[i, :, :] = smoothing_rauch(muf[i, :], Sigf[i, :, :], mus[i + 1, :], Sigs[i + 1, :, :], Xtplus[i], mutilde[i], R, diffusion_coeffs, dim_x, dim_h)
        # except np.linalg.LinAlgError as e:
        #     print(i, muf[i, :], Sigf[i, :, :], mus[i + 1, :], Sigs[i + 1, :, :], Xtplus[i], mutilde[i], self.friction_coeffs[:, self.dim_x :], self.diffusion_coeffs)
        #     print(repr(e))
        #     raise ValueError
    return muh, Sigh
