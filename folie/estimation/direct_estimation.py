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

    def __init__(self, model):
        super().__init__(model)

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
        extra_kwargs = {}
        for key in ["cells_idx", "loc_x"]:
            if key in data[0]:
                extra_kwargs[key] = np.concatenate([trj[key] for trj in data], axis=0)
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
            self.model.diffusion.fit(X, dx_sq / dt, **extra_kwargs)  # We need to estimate the diffusion first in order to have the prefactor of the bias
            bias = np.concatenate([trj["bias"] for trj in data], axis=0)
            self.model.force.fit(X, bias, y=dx / dt, sample_weight=None, **extra_kwargs)
        else:
            bias = 0.0
            self.model.force.fit(X, dx, sample_weight=None, **extra_kwargs)
        dx -= self.model.force(X, bias, **extra_kwargs) * dt
        if dim <= 1:
            dx_sq = dx**2
        else:
            dx_sq = dx[..., None] * dx[:, None, ...]
        self.model.diffusion.fit(X, dx_sq / dt, **extra_kwargs)
        self.model.fitted_ = True
        return self


class UnderdampedKramersMoyalEstimator(KramersMoyalEstimator):
    r"""Implement method of Brückner and Ronceray to obtain underdamped model

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
        print((V**2).sum(), np.sum(acc[:, 0] * V[:, 0], axis=0))
        print(acc.shape)
        if dim <= 1:
            acc = acc.ravel()
        if dim <= 1:
            acc_sq = acc**2
        else:
            acc_sq = acc[..., None] * acc[:, None, ...]
        self.model.diffusion.fit(X, y=acc_sq * 3 * dt / 2)  # We need to estimate the diffusion first in order to have the prefactor of the bias
        self.model.friction.fit(X, y=self.model.diffusion(X))
        print(self.model.friction.coefficients)
        acc_corr = acc  # - self.model.friction(X) / 6.0
        print(V.shape, (V**2).shape, (acc * V[:, 0]).shape)
        print((V**2).sum(), np.sum(acc * V[:, 0], axis=0))
        self.model.meandispl.fit(X, V, bias, y=acc_corr, sample_weight=None)
        acc -= self.model.force(X)
        print((V**2).sum(), np.sum(acc * V[:, 0], axis=0))
        if dim <= 1:
            acc_sq = acc**2
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
                trj["v"] = (0.5 / trj["dt"]) * (np.roll(diffs, -1, axis=0) + diffs)[1:-1]
            else:
                trj["v"] = trj["v"][1:-1]
            trj["a"] = a[1:-1] / (trj["dt"] ** 2)
            trj["x"] = trj["x"][1:-1]
            if "bias" in trj:
                trj["bias"] = trj["bias"][1:-1]
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
