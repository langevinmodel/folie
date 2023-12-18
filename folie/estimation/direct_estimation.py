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
            raise ValueError("Cannot fit Kramers Moyal if the model is not linear in its parameters")

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


class UnderdampedKramersMoyalEstimator(Estimator):
    r"""Implement method of Brückner and Ronceray to obtain underdamped model

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
            raise ValueError("Cannot fit Kramers Moyal if the model is not linear in its parameters")

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

        # Faire calculer l'accélération sur les trajs comme préprocessing
        self._loop_over_trajs(self._preprocess_traj, data.weights, data)

        acc_coeff, accsq_coeff, gram_a = self._loop_over_trajs(self._compute_acc_projection, data.weights, data, self.model)
        acc_norm_coeff = np.linalg.inv(gram_a) @ acc_coeff
        accsq_norm_coeff = np.linalg.inv(gram_a) @ accsq_coeff

        # Et on applique la correction pour obtenir les forces et la friction
        corr_coeff, gram_c = self._loop_over_trajs(self._compute_Ito_correction, data.weights, data, self.model, accsq_norm_coeff)

        self.model.force_coeff = acc_norm_coeff - np.linalg.inv(gram_c) @ corr_coeff
        # D'abord la diffusion
        diffusion_coeff, gram_d = self._loop_over_trajs(self._compute_diffusion, data.weights, data, self.model)
        self.model.coefficients_diffusion = np.linalg.inv(gram_d) @ diffusion_coeff

        # Puis on calcul la moyenne de l'accélération
        current_coeff, gram_d = self._loop_over_trajs(self._compute_current, data.weights, data, self.model)

        self.model.fitted_ = True

        return self

    @staticmethod
    def _preprocess_traj(weight, trj):
        """
        Compute velocity and acceleration
        """

        if "v" not in list(trj.keys()) and "a" not in list(trj.keys()):
            diffs = trj["x"] - np.roll(trj["x"], 1, axis=0)
            a = np.roll(diffs, -1, axis=0) - diffs
            trj["v"] = (0.5 / trj["dt"]) * (np.roll(diffs, -1, axis=0) + diffs)[1:-1]
            trj["a"] = a[1:-1] / (trj["dt"] ** 2)
        return (0,)

    @staticmethod
    def _compute_acc_projection(weight, trj, model):
        """
        Force estimation over one trajectory
        TODO: Adapter vérifier dimension array
        """
        force_basis = model.force_jac_coeffs(trj["x"])  # En vrai ça devrait être base force + friction
        friction_basis = model.force_jac_coeffs()
        # Do concatenation of the basis
        basis = np.concatenate((force_basis, np.dot(trj["v"], friction_basis)))
        return np.dot(basis.T, trj["a"]) / trj["dt"], np.dot(basis.T, trj["a"] ** 2) / trj["dt"], np.dot(basis.T, basis)

    @staticmethod
    def _compute_diffusion(weight, trj, model):
        """
        Force estimation over one trajectory
        TODO: Adapter underdamped
        """

        diffusion_basis = model.diffusion_jac_coeffs(trj["x"])
        return np.dot(diffusion_basis.T, trj["a"] ** 2) * 0.75 * trj["dt"], np.dot(diffusion_basis.T, diffusion_basis)

    @staticmethod
    def _compute_Ito_correction(weight, trj, model, D_coeffs):
        """
        Force estimation over one trajectory
        TODO: Adapter underdamped
        """
