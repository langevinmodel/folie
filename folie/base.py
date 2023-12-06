"""
The code in this file is adapted from deeptime (https://github.com/deeptime-ml/deeptime/blob/main/deeptime/base.py)
"""

import abc
from collections import defaultdict
from inspect import signature
from typing import Optional
from pprint import PrettyPrinter


class _BaseMethodsMixin(abc.ABC):
    """Defines common methods used by both Estimator and Model classes. These are mostly static and low-level
    checking of conformity with respect to conventions.
    """

    def __repr__(self):
        pp = PrettyPrinter(indent=1, depth=2)
        name = "{cls}-{id}:".format(id=id(self), cls=self.__class__.__name__)
        offset = "".join([" "] * len(name))
        params = pp.pformat(self.get_params())
        params = params.replace("\n", "\n" + offset)
        return "{name}[{params}]".format(name=name, params=params)

    def get_params(self, deep=False):
        r"""Get the parameters.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError("class should always " "specify their parameters in the signature" " of their __init__ (no varargs)." " %s doesn't follow this convention." % (cls,))
        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError("Invalid parameter %s for estimator %s. " "Check the list of available parameters " "with `estimator.get_params().keys()`." % (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__

        if type(self).__module__.startswith("floppyMD."):
            from floppyMD import __version__

            return dict(state.items(), _floppyMD_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        from floppyMD import __version__

        if type(self).__module__.startswith("floppyMD."):
            pickle_version = state.pop("_floppyMD_version", None)
            if pickle_version != __version__:
                import warnings

                warnings.warn("Trying to unpickle estimator {0} from version {1} when " "using version {2}. This might lead to breaking code or " "invalid results. Use at your own risk.".format(self.__class__.__name__, pickle_version, __version__), UserWarning)
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)


class Model(_BaseMethodsMixin):
    r"""The model superclass."""

    def copy(self) -> "Model":
        r"""Makes a deep copy of this model.

        Returns
        -------
        copy
            A new copy of this model.
        """
        import copy

        return copy.deepcopy(self)

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self._coefficients = vals

    @property
    def is_linear(self) -> bool:
        """Return True is the model is linear in its parameters"""
        return False


class Estimator(_BaseMethodsMixin):
    r"""Base class of all estimators

    Parameters
    ----------
    model : Model, optional, default=None
        A model which can be used for initialization. In case an estimator is capable of online learning, i.e.,
        capable of updating models, this can be used to resume the estimation process.
    """

    def __init__(self, model=None, n_jobs=1):
        self._model = model
        self.n_jobs = int(n_jobs)
        if self.n_jobs <= 1:
            self._loop_over_trajs = self._loop_over_trajs_serial
        else:
            self._loop_over_trajs = self._loop_over_trajs_parallel

    @abc.abstractmethod
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

    def fetch_model(self) -> Optional[Model]:
        r"""Yields the estimated model. Can be None if :meth:`fit` was not called.

        Returns
        -------
        model : Model or None
            The estimated model or None.
        """
        return self._model

    def fit_fetch(self, data, **kwargs):
        r"""Fits the internal model on data and subsequently fetches it in one call.

        Parameters
        ----------
        data : array_like
            Data that is used to fit the model.
        **kwargs
            Additional arguments to :meth:`fit`.

        Returns
        -------
        model
            The estimated model.
        """
        self.fit(data, **kwargs)
        return self.fetch_model()

    def _loop_over_trajs_serial(self, func, weights, data, *args, **kwargs):
        """
        A generator for iteration over trajectories
        """
        # Et voir alors pour faire une version parallélisé (en distribué)
        array_res = [func(weight, trj, *args, **kwargs) for weight, trj in zip(weights, data)]
        res = [0.0] * len(array_res[0])
        weightsum = weights.sum()
        for weight, single_res in zip(weights, array_res):
            for i, arr in enumerate(single_res):
                res[i] += arr * weight / weightsum
        return res

    def _loop_over_trajs_parallel(self, func, weights, data, *args, **kwargs):
        """
        A generator for iteration over trajectories
        """
        from joblib import Parallel, delayed

        array_res = Parallel(n_jobs=self.n_jobs)(delayed(func)(weight, trj, *args, **kwargs) for weight, trj in zip(weights, data))
        res = [0.0] * len(array_res[0])
        weightsum = weights.sum()
        for weight, single_res in zip(weights, array_res):
            for i, arr in enumerate(single_res):
                res[i] += arr * weight / weightsum
        return res

    @property
    def model(self):
        """Shortcut to :meth:`fetch_model`."""
        return self.fetch_model()

    @property
    def has_model(self) -> bool:
        r"""Property reporting whether this estimator contains an estimated model. This assumes that the model
        is initialized with `None` otherwise.

        :type: bool
        """
        return self._model is not None
