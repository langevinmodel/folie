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


    def _least_square_loss(self,coefficients, fitted_part, data,  **kwargs):
        # print(self._loop_over_trajs(fitted_part, data.weights, data, coefficients, **kwargs))
        return np.sqrt(self._loop_over_trajs(fitted_part, data.weights, data, coefficients, **kwargs))
    
    def _diffusion_loss(self,weight, trj, coefficients,remove_drift=False):
        self.model.diffusion.coefficients=coefficients
        return np.sum(self._diffusion_residuals(**trj,remove_drift=remove_drift))/weight
       
    def _diffusion_residuals(self,xt,x,dt,bias=0.0, remove_drift=False,  **kwargs):
        dim=x.shape[1]
        if dim <= 1:
            dx=xt.ravel()-x.ravel()
        else:
            dx=xt-x
        
        if remove_drift:
            dx= dx-self.model.drift(x, bias, **kwargs) * dt
        
        if dim <= 1:
            dx_sq = dx**2
            return (0.5 * dx_sq / dt-self.model.diffusion(x))**2
        else:
            dx_sq = dx[..., None] * dx[:, None, ...]
            residuals= (0.5 * dx_sq / dt- self.model.diffusion(x))
            return np.einsum("ikl,ikl->i",residuals,residuals)

    def _drift_loss(self,weight, trj, coefficients):
        self.model.pos_drift.coefficients=coefficients
        return np.sum(self._drift_residuals(**trj))/weight
    
    def _drift_residuals(self,xt,x,dt,bias=0.0, **kwargs):
        
        dim=x.shape[1]
        if dim <= 1:
            return ((xt.ravel()-x.ravel())/dt-self.model.drift(x, bias, **kwargs))**2
        else:
            return ((xt-x)/dt-self.model.drift(x, bias, **kwargs))**2

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
        if self.model.is_biased:  # If bias, first determine diffusion
            res=scipy.minimize.least_squares(self._least_square_loss, self.model.diffusion.coefficients,jac= jacobian(self._least_square_loss), args=(self._diffusion_loss,data),kwargs={"remove_drift":False})
            self.model.diffusion.coefficients = res.x
        
        res=scipy.optimize.least_squares(self._least_square_loss, self.model.pos_drift.coefficients,jac= jacobian(self._least_square_loss), args=(self._drift_loss,data))
        self.model.pos_drift.coefficients = res.x

        # Determine diffusion with drift removed
        res=scipy.optimize.least_squares(self._least_square_loss, self.model.diffusion.coefficients,jac= jacobian(self._least_square_loss), args=(self._diffusion_loss,data),kwargs={"remove_drift":True})
        self.model.diffusion.coefficients = res.x
        
        
        self.model.fitted_ = True
        return self
