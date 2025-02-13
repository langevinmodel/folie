from .._numpy import np, jacobian


from ..base import Estimator
import scipy.optimize


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


    def _least_square_loss(self,coefficients,fitted_part,data,  **kwargs):
        return np.sqrt(self._loop_over_trajs(fitted_part, data.weights, data, coefficients, **kwargs))


    def fit(self, data, **kwargs):
        r"""Fits data to the estimator's internal :class:`Model` and overwrites it. This way, every call to
        :meth:`fetch_model` yields an autonomous model instance.

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

        # def diffusion_fit(weight, trj, coefficients):
        #     self.model.diffusion.coefficients=coefficients
        #     dx=trj["xt"] - trj["x"]
        #     if dim <= 1:
        #         dx_sq = dx**2
        #         return np.sum((0.5 * dx_sq / trj["dt"]-self.model.diffusion(trj["x"]))**2)
        #     else:
        #         dx_sq = dx[..., None] * dx[:, None, ...]
        #         residuals= (0.5 * dx_sq / trj["dt"]-self.model.diffusion(trj["x"])).reshape(trj["x"].shape[0],-1)
        #         return np.einsum("ik,ik->i",residuals,residuals)

        # res=scipy.optimize.least_squares(self._least_square_loss, self.coefficients,jac= jacobian(self._least_square_loss), args=(diffusion_fit,data))
        # self.model.diffusion.coefficients = res.x



        # X = np.concatenate([trj["x"] for trj in data], axis=0)
        # for key in ["cells_idx", "loc_x"]:
        #     if key in data[0]:
        #         kwargs[key] = np.concatenate([trj[key] for trj in data], axis=0)
        # # Take weight into account as well
        # dim = X.shape[1]
        # dx = np.concatenate([(trj["xt"] - trj["x"]) for trj in data], axis=0)
        # if dim <= 1:
        #     dx = dx.ravel()
        # # weights = np.concatenate([trj["weight"] for trj in data], axis=0)  # TODO: implement correctly the weights
        # if self.model.is_biased:  # If bias
        #     if dim <= 1:
        #         dx_sq = dx**2
        #     else:
        #         dx_sq = dx[..., None] * dx[:, None, ...]
        #     self.model.diffusion.fit(X, y=0.5 * dx_sq / dt, **kwargs)  # We need to estimate the diffusion first in order to have the prefactor of the bias
        #     bias = np.concatenate([trj["bias"] for trj in data], axis=0)
        #     bias_drift = np.einsum("t...h,th-> t...", self.model.diffusion(X, **kwargs).reshape((*dx.shape, bias.shape[1])), bias)
        #     self.model.drift.fit(X, bias, y=dx / dt - bias_drift, sample_weight=None,  **kwargs)
        # else:
        #     bias = 0.0
        #     self.model.drift.fit(X, y=dx / dt, sample_weight=None,  **kwargs)
        # # print(self.model.drift.coefficients)
        # dx -= self.model.drift(X, bias, **kwargs) * dt
        # if dim <= 1:
        #     dx_sq = dx**2
        # else:
        #     dx_sq = dx[..., None] * dx[:, None, ...]
        # self.model.diffusion.fit(X, y=0.5 * dx_sq / dt, **kwargs)
        self.model.fitted_ = True
        return self
