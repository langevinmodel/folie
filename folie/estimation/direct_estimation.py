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
            self.model.diffusion.fit(X, y=dx_sq / dt, **kwargs)  # We need to estimate the diffusion first in order to have the prefactor of the bias
            bias = np.concatenate([trj["bias"] for trj in data], axis=0)
            bias_force = np.einsum("t...h,th-> t...", self.model.diffusion(X, **kwargs).reshape((*dx.shape, bias.shape[1])), bias)
            self.model.meandispl.fit(X, bias, y=dx / dt - bias_force, sample_weight=None, estimator=estimator, **kwargs)
        else:
            bias = 0.0
            self.model.meandispl.fit(X, y=dx / dt, sample_weight=None, estimator=estimator, **kwargs)
        # print(self.model.meandispl.coefficients)
        dx -= self.model.meandispl(X, bias, **kwargs) * dt
        if dim <= 1:
            dx_sq = dx**2
        else:
            dx_sq = dx[..., None] * dx[:, None, ...]
        self.model.diffusion.fit(X, y=dx_sq / dt, estimator=estimator, **kwargs)
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

    def fit(self, data, **kwargs):
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
        V = np.concatenate([trj["v"] for trj in data], axis=0)
        bias = np.concatenate([trj["bias"] for trj in data], axis=0)
        extra_kwargs = {}
        for key in ["cells_idx", "loc_x"]:
            if key in data[0]:
                extra_kwargs[key] = np.concatenate([trj[key] for trj in data], axis=0)

        # weights = np.concatenate(data.weights, axis=0)  # TODO: implement correctly the weights

        # Take weight into account as well
        dim = X.shape[1]
        acc = np.concatenate([trj["a"] for trj in data], axis=0)
        if dim <= 1:
            acc = acc.ravel()
        self.model.meandispl.fit(X, V, bias, y=acc, sample_weight=None)
        acc -= self.model.force(X)
        if dim <= 1:
            acc_sq = acc ** 2
        else:
            acc_sq = acc[..., None] * acc[:, None, ...]
        self.model.diffusion.fit(X, y=acc_sq * dt)
        self.model.fitted_ = True

        return self

    def preprocess_traj(self, trj, **kwargs):
        """
        Compute velocity and acceleration
        """

        if "a" not in list(trj.keys()):
            diffs = trj["x"] - np.roll(trj["x"], 1, axis=0)
            a = np.roll(diffs, -1, axis=0) - diffs
            if "v" not in list(trj.keys()):
                trj["v"] = (0.5 / trj["dt"]) * (trj["x"] - np.roll(trj["x"], 2, axis=0))[2:-1]
            else:
                trj["v"] = trj["v"][2:-1]
            trj["a"] = a[2:-1] / (trj["dt"] ** 2)
            trj["x"] = trj["x"][2:-1]
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
            self._model.preprocess_traj(trj, **kwargs)
        return trj
