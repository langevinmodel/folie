"""
The code in this file was originnaly adapted from pymle (https://github.com/jkirkby3/pymle)
"""

from .._numpy import np
import warnings
from time import time
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning


from ..base import Estimator
from .direct_estimation import KramersMoyalEstimator, UnderdampedKramersMoyalEstimator
from ..models import BaseModelOverdamped


class EstimatedResult(object):
    def __init__(self, coefficients: np.ndarray, log_like: float, sample_size: int):
        """
        Container for the result of estimation
        :param coefficients: array, the estimated (optimal) coefficients
        :param log_like: float, the final log-likelihood value (at optimum)
        :param sample_size: int, the size of sample used in estimation (don't include S0)
        """
        self.coefficients = coefficients
        self.log_like = log_like
        self.sample_size = sample_size

    @property
    def likelihood(self) -> float:
        """The likelihood with estimated coefficients"""
        return np.exp(self.log_like)

    @property
    def aic(self) -> float:
        """The AIC (Aikake Information Criteria) with estimated coefficients"""
        return 2 * (len(self.coefficients) - self.log_like)

    @property
    def bic(self) -> float:
        """The BIC (Bayesian Information Criteria) with estimated coefficients"""
        return len(self.coefficients) * np.log(self.sample_size) - 2 * self.log_like

    def __str__(self):
        """String representation of the class (for pretty printing the results)"""
        return f"\ncoefficients      | {self.coefficients} \n" f"sample size | {self.sample_size} \n" f"likelihood  | {self.log_like} \n" f"AIC         | {self.aic}\n" f"BIC         | {self.bic}"


class LikelihoodEstimator(Estimator):
    r"""Likelihood-based estimator

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, transition, **kwargs):
        super().__init__(transition.model)
        self.transition = transition

    def fit(self, data, minimizer=None, coefficients0=None, use_jac=True, callback=None, minimize_kwargs={"method": "L-BFGS-B"}, **kwargs):
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
            self.transition.preprocess_traj(trj)
        if coefficients0 is None:
            # TODO, check depending of the order of the model
            if isinstance(self.model, BaseModelOverdamped):
                KramersMoyalEstimator(self.model).fit(data)
            coefficients0 = self.model.coefficients
        if minimizer is None:
            coefficients0 = np.asarray(coefficients0)
            minimizer = minimize

        # Run once, to determine if there is a Jacobian and eventual compilation if needed by numba
        init_val = self._loop_over_trajs(self.transition, data.weights, data, coefficients0, **kwargs)

        if len(init_val) >= 2 and use_jac:
            res = minimizer(self._log_likelihood_negative_with_jac, coefficients0, args=(data,), jac=True, **minimize_kwargs)
        else:
            self.transition.use_jac = False
            res = minimizer(self._log_likelihood_negative, coefficients0, args=(data,), callback=callback, **minimize_kwargs)
        coefficients = res.x

        final_like = -res.fun

        self.model.coefficients = coefficients
        self.model.fitted_ = True

        self.results_ = EstimatedResult(coefficients=coefficients, log_like=final_like, sample_size=data.nobs - 1)

        return self

    def train(self):
        model = self.model
        device = self.device

        sampler = RandomSampler(
            train_data,
            replacement=True,
            num_samples=self.samples_per_ep,
        )
        train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=sampler)
        # trouver comment passer les données
        print("====== Training ======")
        print(f"# epochs: {self.num_epochs}")
        print(f"# examples: {len(train_data)}")
        print(f"# samples used per epoch: {self.samples_per_ep}")
        print(f"batch size: {self.batch_size}")
        print(f"# steps: {len(train_loader)}")
        loss_history = []
        model.train()
        model.to(device)

        # Resume
        last_ckpt_dir = self.get_last_ckpt_dir()
        if last_ckpt_dir is not None:
            print(f"Resuming from {last_ckpt_dir}")
            model.load_state_dict(torch.load(last_ckpt_dir / "ckpt.pt"))
            self.optimizer.load_state_dict(torch.load(last_ckpt_dir / "optimizer.pt"))
            self.lr_scheduler.load_state_dict(torch.load(last_ckpt_dir / "lr_scheduler.pt"))
            ep = int(last_ckpt_dir.name.split("-")[-1]) + 1
        else:
            ep = 0

        train_start_time = time()
        while ep < self.num_epochs:
            print(f"====== Epoch {ep} ======")
            for step, batch in enumerate(train_loader):
                inputs = {k: t.to(device) for k, t in batch.items()}

                # Forward
                outputs = model(**inputs)  # Ici ca doit être la Probability transition
                loss = outputs["loss"]
                loss_history.append(loss.item())

                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if step % self.log_interval == 0:
                    print(
                        {
                            "step": step,
                            "loss": round(loss.item(), 6),
                            "lr": round(self.optimizer.param_groups[0]["lr"], 4),
                            "lambda1": round(self.model.lambda1.item(), 4),
                            "lambda2": round(self.model.lambda2.item(), 4),
                            "time": round(time() - train_start_time, 1),
                        }
                    )
            self.lr_scheduler.step()
            self.checkpoint(ep)
            print(f"====== Epoch {ep} done ======")
        print("====== Training done ======")

    def _log_likelihood_negative(self, coefficients, data, **kwargs):
        return self._loop_over_trajs(self.transition, data.weights, data, coefficients, **kwargs)[0]

    def _log_likelihood_negative_with_jac(self, coefficients, data, **kwargs):
        return self._loop_over_trajs(self.transition, data.weights, data, coefficients, **kwargs)
