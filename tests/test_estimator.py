import pytest
import os
import numpy as np
import folie as fl
import dask.array as da
import torch


@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    elif request.param == "torch":
        trj = torch.from_numpy(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    for i in range(1, trj.shape[1]):
        trj_list.append(trj[:, i : (i + 1)])
    trj_list.stats
    return trj_list


@pytest.fixture
def data2d(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    elif request.param == "torch":
        trj = torch.from_numpy(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    trj_list.append(trj[:, 1:3])
    trj_list.stats
    return trj_list


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.Polynomial, {"deg": 3}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
        (fl.functions.Fourier, {"order": 3}),
    ],
)
def test_direct_estimator(data, request, fct, parameters):
    model = fl.models.OverdampedFunctions(fct(**parameters))
    estimator = fl.KramersMoyalEstimator(model)
    model = estimator.fit_fetch(data)
    assert model.fitted_


@pytest.mark.parametrize("data2d", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.Polynomial, {"deg": 3, "polynom": np.polynomial.Chebyshev}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
        (fl.functions.Fourier, {"order": 3}),
    ],
)
def test_direct_estimator2d(data2d, request, fct, parameters):
    model = fl.models.OverdampedFunctions(fct(**parameters), dim=2)
    estimator = fl.KramersMoyalEstimator(model)
    model = estimator.fit_fetch(data2d)
    assert model.fitted_


@pytest.mark.skip(reason="Not implemented yet")
@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
def test_direct_estimator_underdamped(data, request):
    fun_lin = fl.functions.Linear()
    fun_cst = fl.functions.Constant()
    model = fl.models.UnderdampedFunctions(fun_lin, fun_lin.copy(), fun_cst)
    estimator = fl.UnderdampedKramersMoyalEstimator(model)
    model = estimator.fit_fetch(data)
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_likelihood_estimator(data, request):
    fun_lin = fl.functions.Linear()
    model = fl.models.OverdampedFunctions(fun_lin, dim=1)
    estimator = fl.LikelihoodEstimator(fl.EulerDensity(model))
    model = estimator.fit_fetch(data, coefficients0=[1.0, 1.0])
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_numba_likelihood_estimator(data, request):
    n_knots = 20
    epsilon = 1e-10
    model = fl.models.OverdampedFreeEnergy(np.linspace(data.stats.min - epsilon, data.stats.max + epsilon, n_knots), 1.0)
    estimator = fl.LikelihoodEstimator(fl.EulerNumbaOptimizedDensity(model))

    model = estimator.fit_fetch(data, coefficients0=np.concatenate((np.zeros(n_knots), np.zeros(n_knots) + 1.0)))
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_em_estimator(data, request):
    fun_lin = fl.functions.Linear().fit(data)
    fun_cst = fl.functions.Constant().fit(data).resize((3, 3))
    model = fl.models.OverdampedHidden(fun_lin, fun_lin.copy(), fun_cst, dim=1, dim_h=2)
    estimator = fl.EMEstimator(fl.EulerDensity(model), max_iter=3, verbose=3, verbose_interval=1)
    model = estimator.fit_fetch(data)
    assert model.fitted_
