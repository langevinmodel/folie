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

    def fit(self, data, estimator=linear_model.LinearRegression(copy_X=False, fit_intercept=False), density_weighting=None, density_kwargs=None, **kwargs):
        r"""Fits data to the estimator's internal :class:`Model` and overwrites it. This way, every call to
        :meth:`fetch_model` yields an autonomous model instance.

        Parameters
        ----------
        data : array_like
            Data that is used to fit a model.

        estimator: sklearn compatible estimator
            Defaut to sklearn.linear_model.LinearRegression(copy_X=False, fit_intercept=False) but any compatible estimator can be used.
            Estimator should have a coef attibutes after fitting

        density_weighting : str or None, default=None
            Method to estimate point density to discount noisy boundaries.
            - None: No density weighting.
            - 'knn': Fast, continuous estimation using K-Nearest Neighbors (Recommended).
            - 'hist': Lightning-fast O(N) binned estimation (Best for 1D/2D).
            - 'kde': Rigorous but slow Kernel Density Estimation.

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

        if density_weighting is not None:
            density_kwargs = density_kwargs or {}
            data.compute_density(method=density_weighting, **density_kwargs)

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

        # --- Extract Weights ---
        sample_weights_list = []
        for trj in data:
            w = trj.get("weight", np.ones(len(trj["x"])))
            if "density" in trj and density_weighting is not None:
                w = w * trj["density"]
            sample_weights_list.append(w)
        sample_weight = np.concatenate(sample_weights_list, axis=0)

        if self.model.is_biased:  # If bias
            if dim <= 1:
                dx_sq = dx**2
            else:
                dx_sq = dx[..., None] * dx[:, None, ...]
            self.model.diffusion.fit(
                X, y=0.5 * dx_sq / dt, sample_weight=sample_weight, estimator=estimator, **kwargs
            )  # We need to estimate the diffusion first in order to have the prefactor of the bias
            bias = np.concatenate([trj["bias"] for trj in data], axis=0)
            bias_drift = np.einsum("t...h,th-> t...", self.model.diffusion(X, **kwargs).reshape((*dx.shape, bias.shape[1])), bias)
            self.model.drift.fit(X, bias, y=dx / dt - bias_drift, sample_weight=sample_weight, estimator=estimator, **kwargs)
        else:
            bias = 0.0
            self.model.drift.fit(X, y=dx / dt, sample_weight=sample_weight, estimator=estimator, **kwargs)
        # print(self.model.drift.coefficients)
        dx = dx - self.model.drift(X, bias, **kwargs) * dt
        if dim <= 1:
            dx_sq = dx**2
        else:
            dx_sq = dx[..., None] * dx[:, None, ...]
        self.model.diffusion.fit(X, y=0.5 * dx_sq / dt, sample_weight=sample_weight, estimator=estimator, **kwargs)
        self.model.fitted_ = True
        return self
