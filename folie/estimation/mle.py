"""
The code in this file was originnaly adapted from pymle (https://github.com/jkirkby3/pymle)
"""

import numpy as np
import warnings
from time import time
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning


from ..base import Estimator


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


class CallbackFunctor:
    """
    Callback or scipy minimize in order to store history if wanted
    """

    def __init__(self, obj_fun):
        """
        obj_fun is a provided function to extract value from OptimizedResult
        """
        self.history = [np.inf]
        self.sols = []
        self.num_calls = 0
        self.obj_fun = obj_fun

    def __call__(self, x):
        fun_val = self.obj_fun(x)
        self.num_calls += 1
        if fun_val < self.history[-1]:
            self.sols.append(x)
            self.history.append(fun_val)

    def save_sols(self, filename):
        sols = np.array([sol for sol in self.sols])
        np.savetxt(filename, sols)


class LikelihoodEstimator(Estimator):
    r"""Base class of all estimators

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, transition, **kwargs):
        super().__init__(transition.model)
        self.transition = transition

    def fit(self, data, minimizer=None, coefficients0=None, use_jac=True, **kwargs):
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

        if self.transition.do_preprocess_traj:
            for trj in data:
                self.transition.preprocess_traj(trj)
        if coefficients0 is None:
            coefficients0 = self.model.coefficients
            # TODO, use exact optimisation to provide first set of parameters
        if minimizer is None:
            coefficients0 = np.asarray(coefficients0)
            minimizer = minimize

        if self.transition.has_jac and use_jac:
            res = minimizer(self._log_likelihood_negative_with_jac, coefficients0, args=(data,), jac=True, method="L-BFGS-B")
        else:
            res = minimizer(self._log_likelihood_negative, coefficients0, args=(data,), method="L-BFGS-B")
        coefficients = res.x

        final_like = -res.fun

        self.model.coefficients = coefficients
        self.model.fitted_ = True

        self.results_ = EstimatedResult(coefficients=coefficients, log_like=final_like, sample_size=data.nobs - 1)

        return self

    def _log_likelihood_negative(self, coefficients, data, **kwargs):
        return self._loop_over_trajs(self.transition, data.weights, data, coefficients, **kwargs)[0]

    def _log_likelihood_negative_with_jac(self, coefficients, data, **kwargs):
        return self._loop_over_trajs(self.transition, data.weights, data, coefficients, **kwargs)


class ELBOEstimator(LikelihoodEstimator):
    """
    Maximize the Evidence lower bound.
    Similar to EM estimation but the expectation is realized inside the minimization loop
    """

    def __init__(self, model=None, transition=None, **kwargs):
        super().__init__(model)

    def _log_likelihood_negative(self, coefficients, data, **kwargs):
        return self._loop_over_trajs(self.transition.loglikelihood_w_correction, data.weights, data, coefficients, **kwargs)[0]

    def _log_likelihood_negative_with_jac(self, coefficients, data, **kwargs):
        return self._loop_over_trajs(self.transition.loglikelihood_w_correction, data.weights, data, coefficients, **kwargs)


class EMEstimator(LikelihoodEstimator):
    """
    Maximize the likelihood using Expectation-maximization algorithm
    TODO: Replace all history by a callback
    """

    def __init__(
        self,
        model,
        transition,
        *args,
        tol=1e-5,
        max_iter=100,
        n_init=1,
        warm_start=False,
        no_stop=False,
        verbose=0,
        verbose_interval=10,
        **kwargs,
    ):
        super().__init__(model)
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.warm_start = warm_start

        self.no_stop = no_stop

    def fit(self, data, minimizer=None, coefficients0=None, use_jac=True, **kwargs):
        """
        In this do a loop that alternatively minimize and compute expectation
        """

        if self.transition.do_preprocess_traj:
            for trj in data:
                self.transition.preprocess_traj(trj)
        if coefficients0 is None:
            raise NotImplementedError  # We could use then an exact estimation

        if minimizer is None:
            coefficients = np.asarray(coefficients0)
            minimizer = minimize

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        self.logL = np.empty((n_init, self.max_iter))
        self.logL[:] = np.nan
        self.coeffs_list_all = []

        # For referencement
        best_coeffs = None
        best_n_iter = -1
        best_n_init = -1

        for init in range(n_init):
            coeff_list_init = []
            if do_init:
                coefficients = self._initialize_parameters(coefficients0)  # Need to randomize initial coefficients if multiple run
            self._print_verbose_msg_init_beg(init)
            lower_bound = -np.infty if do_init else self.lower_bound_
            lower_bound_m_step = -np.infty
            # Algorithm loop
            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound
                # E step
                new_stat = self.transition.e_step(data)

                lower_bound = -self.transition(new_stat)  # TODO A changer
                if self.verbose >= 2:
                    if lower_bound - lower_bound_m_step < 0:
                        print("Delta ll after E step:", lower_bound - lower_bound_m_step)
                curr_coeffs = self.model.coefficients
                curr_coeffs["ll"] = lower_bound
                coeff_list_init.append(curr_coeffs)
                # M Step
                if self.transition.has_jac and use_jac:
                    res = minimizer(self._log_likelihood_negative_with_jac, coefficients0, args=(data,), jac=True, method="L-BFGS-B")
                else:
                    res = minimizer(self._log_likelihood_negative, coefficients0, args=(data,), method="L-BFGS-B")
                coefficients = res.x

                lower_bound_m_step = -res.fun
                if self.verbose >= 2 and lower_bound_m_step - lower_bound < 0:
                    print("Delta ll after M step:", lower_bound_m_step - lower_bound)
                if np.isnan(lower_bound_m_step) or not np.isfinite(np.sum(coefficients)):  # If we have nan value we simply restart the iteration
                    warnings.warn(
                        "Initialization %d has NaN values. Ends iteration" % (init),
                        ConvergenceWarning,
                    )
                    if self.verbose >= 2:
                        print(self.model.coefficients)
                        print("ll: {}".format(lower_bound))
                    break

                self.logL[init, n_iter - 1] = lower_bound
                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change, lower_bound)

                if lower_bound > max_lower_bound:
                    max_lower_bound = lower_bound
                    best_coeffs = self.model.coefficients
                    best_n_iter = n_iter
                    best_n_init = init

                if abs(change) < self.tol:
                    self.converged_ = True
                    if not self.no_stop:
                        break

            self._print_verbose_msg_init_end(lower_bound, n_iter)
            self.coeffs_list_all.append(coeff_list_init)
            if not self.converged_:
                warnings.warn(
                    "Initialization %d did not converge. " "Try different init parameters, " "or increase max_iter, tol " "or check for degenerate data." % (init + 1),
                    ConvergenceWarning,
                )
        if best_coeffs is not None:
            self.model.coefficients = best_coeffs
        self.n_iter_ = best_n_iter
        self.n_best_init_ = best_n_init
        self.lower_bound_ = max_lower_bound
        self._print_verbose_msg_fit_end(max_lower_bound, best_n_init, best_n_iter)

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time
        if self.verbose >= 3:
            print("----------------Current parameters values------------------")
            print(self.model.coefficients)

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll, log_likelihood):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("***Iteration EM*** : {} / {} --- Current loglikelihood {}".format(n_iter, self.max_iter, log_likelihood))
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "***Iteration EM*** :%d / %d\t time lapse %.5fs\t Current loglikelihood %.5f loglikelihood change %.5f"
                    % (
                        n_iter,
                        self.max_iter,
                        cur_time - self._iter_prev_time,
                        log_likelihood,
                        diff_ll,
                    )
                )
                self._iter_prev_time = cur_time
            if self.verbose >= 3:
                print("----------------Current parameters values------------------")
                print(self.model.coefficients)

    def _print_verbose_msg_init_end(self, ll, best_iter):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s at step %i \t ll %.5f" % (self.converged_, best_iter, ll))
        elif self.verbose >= 2:
            print("Initialization converged: %s at step %i \t time lapse %.5fs\t ll %.5f" % (self.converged_, best_iter, time() - self._init_prev_time, ll))
            print("----------------Current parameters values------------------")
            print(self.model.coefficients)

    def _print_verbose_msg_fit_end(self, ll, best_init, best_iter):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Fit converged: %s Init: %s at step %i \t ll %.5f" % (self.converged_, best_init, best_iter, ll))
        elif self.verbose >= 2:
            print(
                "Fit converged: %s Init: %s at step %i \t time lapse %.5fs\t ll %.5f"
                % (
                    self.converged_,
                    best_init,
                    best_iter,
                    time() - self._init_prev_time,
                    ll,
                )
            )
            print("----------------Fitted parameters values------------------")
            print(self.model.coefficients)
