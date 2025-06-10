from .._numpy import np


from ..base import Estimator
from sklearn import linear_model


class KramersMoyalEstimator(Estimator):
    r"""Kramers-Moyal estimator

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def preprocess_traj(self, trj, **kwargs):
        """
        Basic preprocessing
        """
        if "xt" not in trj:  # ie, not preprocessing yet
            trj["xt"] = trj["x"][1:]
            trj["x"] = trj["x"][:-1]
            if "bias" in trj:
                trj["bias"] = trj["bias"][:-1]
            else:
                trj["bias"] = np.zeros((1, trj["x"].shape[1]))
            if hasattr(self._model, "dim_h"):
                if self._model.dim_h > 0:
                    trj["sig_h"] = np.zeros((trj["x"].shape[0], 2 * self._model.dim_h, 2 * self._model.dim_h))
                    trj["x"] = np.concatenate((trj["x"], np.zeros((trj["x"].shape[0], self._model.dim_h))), axis=1)
                    trj["xt"] = np.concatenate((trj["xt"], np.zeros((trj["xt"].shape[0], self._model.dim_h))), axis=1)
                    trj["bias"] = np.concatenate((trj["bias"], np.zeros((trj["bias"].shape[0], self._model.dim_h))), axis=1)
            self._model.preprocess_traj(trj, **kwargs)
        return trj

    def fit(self, data, estimator=linear_model.LinearRegression(copy_X=False, fit_intercept=False), **kwargs):
        r"""Fits data to the estimator's internal :class:`Model` and overwrites it. This way, every call to
        :meth:`fetch_model` yields an autonomous model instance.

        Parameters
        ----------
        data : array_like
            Data that is used to fit a model.

        estimator: sklearn compatible estimator
            Defaut to sklearn.linear_model.LinearRegression(copy_X=False, fit_intercept=False) but any compatible estimator can be used.
            Estimator should have a coef attibutes after fitting

        **kwargs
            Additional kwargs.

        Returns
        -------
        self : Estimator
            Reference to self.
        """
        estimator.n_jobs = self.n_jobs
        for trj in data:
            self.preprocess_traj(trj)

        dt = data[0]["dt"]

        X = np.concatenate([trj["x"] for trj in data], axis=0)
        for key in ["cells_idx", "loc_x"]:
            if key in data[0]:
                kwargs[key] = np.concatenate([trj[key] for trj in data], axis=0)
        # Take weight into account as well
        dim = X.shape[1]
        dx = np.concatenate([(trj["xt"] - trj["x"]) for trj in data], axis=0)
        if dim <= 1:
            dx = dx.ravel()
        # weights = np.concatenate([trj["weight"] for trj in data], axis=0)  # TODO: implement correctly the weights
        if self.model.is_biased:  # If bias
            if dim <= 1:
                dx_sq = dx**2
            else:
                dx_sq = dx[..., None] * dx[:, None, ...]
            self.model.diffusion.fit(X, y=0.5 * dx_sq / dt, **kwargs)  # We need to estimate the diffusion first in order to have the prefactor of the bias
            bias = np.concatenate([trj["bias"] for trj in data], axis=0)
            bias_drift = np.einsum("t...h,th-> t...", self.model.diffusion(X, **kwargs).reshape((*dx.shape, bias.shape[1])), bias)
            self.model.drift.fit(X, bias, y=dx / dt - bias_drift, sample_weight=None, estimator=estimator, **kwargs)
        else:
            bias = 0.0
            self.model.drift.fit(X, y=dx / dt, sample_weight=None, estimator=estimator, **kwargs)
        # print(self.model.drift.coefficients)
        dx -= self.model.drift(X, bias, **kwargs) * dt
        if dim <= 1:
            dx_sq = dx**2
        else:
            dx_sq = dx[..., None] * dx[:, None, ...]
        self.model.diffusion.fit(X, y=0.5 * dx_sq / dt, estimator=estimator, **kwargs)
        self.model.fitted_ = True
        return self


class UnderdampedKramersMoyalEstimator(KramersMoyalEstimator):
    r"""Obtain underdamped model. It's a biased estimator that does not yield correct results but still provide interesting starting point for optimisation

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, model):
        super().__init__(model)

    def fit(self, data, correct_finite_diff_vel=True, **kwargs):
        r"""Fits data to the estimator's internal :class:`Model` and overwrites it. This way, every call to
        :meth:`fetch_model` yields an autonomous model instance. Sometimes a :code:`partial_fit` method is available,
        in which case the model can get updated by the estimator.

        Parameters
        ----------
        data : array_like
            Data that is used to fit a model.
        **kwargs
            Additional kwargs.

        Returns
        -------
        self : Estimator
            Reference to self.
        """

        for trj in data:
            self.preprocess_traj(trj)

        dt = data[0]["dt"]

        X = np.concatenate([trj["x"] for trj in data], axis=0)
        U = np.concatenate([trj["u"] for trj in data], axis=0)
        V = np.concatenate([trj["v"] for trj in data], axis=0)
        
        bias = np.concatenate([trj["bias"] for trj in data], axis=0)
        extra_kwargs = {}
        for key in ["cells_idx", "loc_x"]:
            if key in data[0]:
                extra_kwargs[key] = np.concatenate([trj[key] for trj in data], axis=0)

        # ----- Initial fit (works if velocities are exact) -----
        dim = X.shape[1]
        acc = np.concatenate([trj["a"] for trj in data], axis=0)
        if dim <= 1:
            acc = acc.ravel()
        # fit drift to average acceleration
        self.model.drift.fit(X, V, bias, y=acc, sample_weight=None)
        # subtracting mean force has negligible impact as acceleration
        # noise terms diverge ( O(dt^-1/2) vs. O(1) drift terms ).
        acc -= self.model.drift(X,V)
        if dim <= 1:
            acc_sq = acc ** 2
        else:
        # fit diffusion to squared acceleration
            acc_sq = acc[..., None] * acc[:, None, ...]
        self.model.diffusion.fit(X, y=acc_sq * dt / 2)

        # ------ Finite difference velocities correction ------
        # test if velocities were computed by finite differences
        if correct_finite_diff_vel and np.array_equal(U, V):
            # multiply diffusion coefficients by 3/2 factor
            self.model.diffusion.coefficients *= 3/2
            # compute inverse square root Gram matrix of the basis of functions
            # Fx's columns are vectors of basis functions evaluated over the trajectory
            Fx = self.model.drift.grad_coeffs(X,V).reshape((X.shape[0] * self.model.drift.output_size_, -1))
            H = inverse_sqrt_gram(Fx)
            # compute the correction in the original basis
            # switch to orthonormalized basis (right-side product by H)
            # switch back to original basis (left-side product by H.T=H)
            corr = H.T @ self.compute_drift_correction(X) @ H
            # add bias correction to initial coefficients
            self.model.drift.coefficients += corr
          
        self.model.fitted_ = True
        return self

    def preprocess_traj(self, trj, **kwargs):
        """
        Compute velocity and acceleration
        """

        if "u" not in list(trj.keys()):
            # u_n = u_n+1 - u_n-1 / 2dt
            trj["u"] = (0.5 / trj["dt"]) * (trj["x"] - np.roll(trj["x"], 2, axis=0))
            trj["u"] = np.roll(trj["u"], -1, axis=0)

        if "v" not in trj:
            trj["v"] = trj["u"].copy()
            if "a" not in trj:
                # If we want to retrieve the 12/5 correction, acceleration
                # must be defined as v_n+1 - v_n / dt
                # trj["a"] = ( trj["v"] - np.roll(trj["v"], 1, axis=0) )[2:-1] / trj["dt"]
                diffs = trj["x"] - np.roll(trj["x"], 1, axis=0)
                a = np.roll(diffs, -1, axis=0) - diffs
                trj["a"] = a[1:-2] / (trj["dt"] ** 2)
        elif "a" not in trj:
            # Purposefully define the acceleration as v_n+1 - v_n / dt
            # for further fitting
            trj["a"] = ( trj["v"] - np.roll(trj["v"], 1, axis=0) )[2:-1] / trj["dt"]
            # trj["a"] = (0.5 / trj["dt"]) * (trj["v"] - np.roll(trj["v"], 2, axis=0))
            # trj["a"] = np.roll(trj["a"], -1, axis=0)[1:-2]


        if "vt" not in trj:
            trj["vt"] = trj["v"][2:-1]
            trj["v"] = trj["v"][1:-2]
            trj["u"] = trj["u"][1:-2]
            
        if "xt" not in trj:
            trj["xt"] = trj["x"][2:-1]
            trj["x"] = trj["x"][1:-2]

        if "bias" in trj:
            trj["bias"] = trj["bias"][2:-1]
        else:
            trj["bias"] = np.zeros((1, trj["x"].shape[1]))

        if hasattr(self._model, "dim_h"):
            if self._model.dim_h > 0:
                trj["sig_h"] = np.zeros((trj["x"].shape[0], 2 * self._model.dim_h, 2 * self._model.dim_h))
                trj["v"] = np.concatenate((trj["v"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["a"] = np.concatenate((trj["a"], np.zeros((trj["a"].shape[0], self._model.dim_h))), axis=1)
                trj["bias"] = np.concatenate((trj["bias"], np.zeros((trj["bias"].shape[0], self._model.dim_h))), axis=1)
                trj["vt"] = np.concatenate((trj["vt"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)
                trj["x"] = np.concatenate((trj["x"], np.zeros((trj["x"].shape[0], self._model.dim_h))), axis=1)
                trj["xt"] = np.concatenate((trj["xt"], np.zeros((trj["v"].shape[0], self._model.dim_h))), axis=1)

        self._model.preprocess_traj(trj, **kwargs)
        return trj

    def compute_drift_correction(self, X):
        r"""Compute drift correction due to finite-difference velocity 
        estimation in underdamped stochastic systems :footcite:`bruckner2020inferring`.
    
        This correction removes the :math:`O(dt^0)` bias introduced by the linear 
        regression fit of the drift function on finite-difference velocities arising
        from the product of :math:`O(dt^{-1/2})` accelerations with :math:`O(dt^{1/2})`
        random noise difference between exact and esimated velocities.
    
        Parameters
        ----------
        X : ndarray of shape (T, D)
            Input trajectory data (e.g., positions), where T is the number of
            time steps and D is the dimensionality of the system.
    
        Returns
        -------
        correction_vector : ndarray of shape (N,)
            Concatenated correction to the drift coefficients. The first portion 
            (corresponding to free-energy contributions) is zero, while the 
            second part contains the corrected velocity-dependent friction terms.
    
        References
        ----------
        .. footbibliography::
        """
        diff = self.model.diffusion(X)
        init_coeffs = self.model.friction.coefficients

        # Fit friction function on diffusion
        self.model.friction.fit(X, y=diff)
        # Compute Gram matrix of friction base functions
        Gx = self.model.friction.grad_coeffs(X).reshape((X.shape[0] * self.model.friction.output_size_, -1))
        Gram_gamma = build_gram_matrix(Gx)
        # Cancel inverse Gram matrix product from linear regression
        # fit by multiplying by Gram matrix on the left
        correction = Gram_gamma @ self.model.friction.coefficients
        # Re-assign initial friction coefficients
        self.model.friction.coefficients = init_coeffs
        
        pos_drift_n_funcs = len(self.model.pos_drift.coefficients)
        friction_n_funcs = len(self.model.friction.coefficients)
        # Free-energy terms have no velocity dependence so corresponding
        # correction is zero. Output has shape (len(model.drift.coefficients),).
        return np.concatenate( ( np.zeros(pos_drift_n_funcs), correction ) , axis=-1 ) 

    def add_noise_to_correct_velocities(self, trj):
        r""" Add taylored noise to finite-difference velocities to correct drift fitting.
                
        Parameters
        ----------
        trj : dict
            Trajectory dictionary containing positions, velocities and other data.
    
        Returns
        -------
        trj : dict
            Updated trajectory with corrected velocities.
        """
        # Compute sigma squared from average diffusion
        sigma_sq = 2 * self.model.diffusion(trj["x"]).ravel().mean() * trj["dt"]
        
        # Correct velocity by adding two gaussian white noises
        # with fine-tuned coefficients a and b that depend on sigma_sq
        a = 1/4*(1+np.sqrt(5/3)) * np.sqrt(sigma_sq)
        b = 1/4*(-1+np.sqrt(5/3)) * np.sqrt(sigma_sq)
        g = np.random.default_rng().standard_normal(size = trj["v"].shape)
        trj["v"] = trj["u"] + a * g + b * np.roll(g, 1, axis=0)

        # Recompute acceleration from new velocities
        trj["a"] = ( trj["v"] - np.roll(trj["v"], 1, axis=0) ) / trj["dt"]
        trj["a"] = np.roll(trj["a"], -1, 0)

        # Cut out last value of acceleration (meaningless)
        # and harmonize all other data arrays.
        for val in trj.values():
            if isinstance(val, np.ndarray):
                val = val[:-1]     
        return trj

def build_gram_matrix(b_eval):
    return b_eval.T @ b_eval / b_eval.shape[0]           

def inverse_sqrt_gram(b_eval, cutoff=1e-10):
    """
    Compute B^{-1/2} using SVD, which is stable even if B is nearly singular.
    """
    U, S, Vh = np.linalg.svd(build_gram_matrix(b_eval))
    inv_sqrt_S = np.diag([1 / np.sqrt(s) if s > cutoff else 0.0 for s in S])
    B_inv_sqrt = U @ inv_sqrt_S @ Vh
    return B_inv_sqrt


class UnderdampedFDTKramersMoyalEstimator(UnderdampedKramersMoyalEstimator):
    def __init__(self, model):
        super().__init__(model)

    def fit(self, data, correct_finite_diff_vel=True, **kwargs):
        r"""Fits data to the estimator's internal :class:`Model` and overwrites it. This way, every call to
        :meth:`fetch_model` yields an autonomous model instance. Sometimes a :code:`partial_fit` method is available,
        in which case the model can get updated by the estimator.

        Parameters
        ----------
        data : array_like
            Data that is used to fit a model.
        **kwargs
            Additional kwargs.

        Returns
        -------
        self : Estimator
            Reference to self.
        """

        for trj in data:
            self.preprocess_traj(trj)

        dt = data[0]["dt"]

        X = np.concatenate([trj["x"] for trj in data], axis=0)
        U = np.concatenate([trj["u"] for trj in data], axis=0)
        V = np.concatenate([trj["v"] for trj in data], axis=0)
        
        bias = np.concatenate([trj["bias"] for trj in data], axis=0)
        extra_kwargs = {}
        for key in ["cells_idx", "loc_x"]:
            if key in data[0]:
                extra_kwargs[key] = np.concatenate([trj[key] for trj in data], axis=0)

        # ----- Initial fit (works if velocities are exact) -----
        dim = X.shape[1]
        acc = np.concatenate([trj["a"] for trj in data], axis=0)
        if dim <= 1:
            acc = acc.ravel()
            acc_sq = acc ** 2
        else:
            acc_sq = acc[..., None] * acc[:, None, ...]
        # fit diffusion to squared acceleration
        self.model.diffusion.fit(X, y=acc_sq * dt / 2)

        # ------ Finite difference velocities correction ------
        # test if velocities were computed by finite differences
        if correct_finite_diff_vel and np.array_equal(U, V):
            # multiply diffusion coefficients by 3/2 factor
            self.model.diffusion.coefficients *= 3/2

        # fit drift to average acceleration
        shape_0 = self.model.pos_drift(X, bias).shape
        self.model.pos_drift.fit(X, bias, y = acc + np.einsum("t...h,th-> t...", self.model.friction(X, bias).reshape((*shape_0, dim)), V), sample_weight=None)
   
        self.model.fitted_ = True
        return self

