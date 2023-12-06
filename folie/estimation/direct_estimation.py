import numpy as np


from ..base import Estimator


class KramersMoyalEstimator(Estimator):
    r"""Base class of all estimators

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, model):
        super().__init__(model)
        # Should check is the model is linear in parameters
        if not self._model.is_linear:
            raise ValueError("Cannot fit Karmers Moyal if the model is not linear in its parameters")

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

        force_coeff, gram_f = self._loop_over_trajs(self._compute_force, data.weights, data, self.model)
        self.model.coefficients_force = np.linalg.inv(gram_f) @ force_coeff

        diffusion_coeff, gram_d = self._loop_over_trajs(self._compute_diffusion, data.weights, data, self.model)
        self.model.coefficients_diffusion = np.linalg.inv(gram_d) @ diffusion_coeff

        self.model.fitted_ = True

        return self

    @staticmethod
    def _compute_force(weight, trj, model):
        """
        Force estimation over one trajectory
        """
        x = trj["x"][:-1]
        dx = trj["x"][1:] - trj["x"][:-1]

        force_basis = model.force_jac_coeffs(x)
        return np.dot(force_basis.T, dx) / trj["dt"], np.dot(force_basis.T, force_basis)

    @staticmethod
    def _compute_diffusion(weight, trj, model):
        """
        Force estimation over one trajectory
        """
        x = trj["x"][:-1]
        dx = trj["x"][1:] - trj["x"][:-1] - model.force(x) * trj["dt"]

        diffusion_basis = model.diffusion_jac_coeffs(x)
        return np.dot(diffusion_basis.T, dx**2) / trj["dt"], np.dot(diffusion_basis.T, diffusion_basis)
