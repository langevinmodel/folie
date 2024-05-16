import pytest
import os
from folie._numpy import np
import folie as fl
import dask.array as da
import torch
import skfem


# TODO: add also xarray and pandas into the data test
@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/datasets/example_2d.trj"))
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
    trj = np.loadtxt(os.path.join(file_dir, "../examples/datasets/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    elif request.param == "torch":
        trj = torch.from_numpy(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    trj_list.append(trj[:, 1:3])
    trj_list.stats
    return trj_list


@pytest.fixture
def data_biased(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/datasets/example_biased_umbrella.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    elif request.param == "torch":
        trj = torch.from_numpy(trj)
    dt = trj[1, 0] - trj[0, 0]
    trj_list = fl.Trajectories()
    trj_list.append(fl.Trajectory(dt, trj[:, 1:2], bias=trj[:, 4:5]))
    trj_list.stats
    return trj_list


@pytest.fixture
def data_short(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/datasets/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    elif request.param == "torch":
        trj = torch.from_numpy(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    for i in range(1, trj.shape[1]):
        trj_list.append(trj[:50, i : (i + 1)])
    trj_list.stats
    return trj_list


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Linear, {}),
        (fl.functions.Polynomial, {"deg": 3}),
    ],
)
def test_direct_estimator(data, request, fct, parameters):
    model = fl.models.Overdamped(fct(**parameters))
    estimator = fl.KramersMoyalEstimator(model)
    model = estimator.fit_fetch(data)
    assert model.fitted_


@pytest.mark.parametrize("data2d", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Linear, {}),
        (fl.functions.Polynomial, {"deg": 3}),
    ],
)
def test_direct_estimator2d(data2d, request, fct, parameters):
    model = fl.models.Overdamped(fct(domain=fl.Domain.Rd(2), **parameters), dim=2)
    estimator = fl.KramersMoyalEstimator(model)
    model = estimator.fit_fetch(data2d)
    assert model.fitted_


@pytest.mark.parametrize("data_biased", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Polynomial, {"deg": 3}),
    ],
)
def test_direct_estimator_biased(data_biased, request, fct, parameters):
    model = fl.models.Overdamped(fct(**parameters), has_bias=True)
    estimator = fl.KramersMoyalEstimator(model)
    model = estimator.fit_fetch(data_biased)
    model.remove_bias()
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_likelihood_estimator(data, request):
    fun_lin = fl.functions.Linear()
    model = fl.models.Overdamped(fun_lin, dim=1)
    estimator = fl.LikelihoodEstimator(fl.EulerDensity(model))
    model = estimator.fit_fetch(data)
    assert model.fitted_


@pytest.mark.parametrize("data2d", ["numpy"], indirect=True)
def test_likelihood_estimator2d(data2d, request):
    fun_lin = fl.functions.Linear(domain=fl.Domain.Rd(2))
    fun_diff = fl.functions.Polynomial(deg=2, domain=fl.Domain.Rd(2))
    model = fl.models.Overdamped(fun_lin, fun_diff, dim=2)
    estimator = fl.LikelihoodEstimator(fl.EulerDensity(model))
    model = estimator.fit_fetch(data2d)
    assert model.fitted_


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_fem_likelihood_estimator(data, request):

    n_knots = 20
    epsilon = 1e-10
    m = skfem.MeshLine(np.linspace(data.stats.min - epsilon, data.stats.max + epsilon, n_knots).ravel())
    domain = fl.MeshedDomain(m)
    fun = fl.functions.FiniteElement(domain, skfem.ElementLineP1())
    model = fl.models.Overdamped(fun, dim=1)
    estimator = fl.LikelihoodEstimator(fl.EulerDensity(model))

    model = estimator.fit_fetch(data)
    assert model.fitted_


@pytest.mark.parametrize("data_biased", ["numpy"], indirect=True)
def test_likelihood_estimator_biased(data_biased, request):
    fun_lin = fl.functions.Linear()
    model = fl.models.Overdamped(fun_lin, dim=1, has_bias=True)
    estimator = fl.LikelihoodEstimator(fl.EulerDensity(model))
    model = estimator.fit_fetch(data_biased)
    model.remove_bias()
    assert model.fitted_


@pytest.mark.skip(reason="A bug remain here, to be inestigated")
@pytest.mark.parametrize("data_short", ["numpy"], indirect=True)
def test_em_estimator(data_short, request):
    fun_lin = fl.functions.Linear()
    fun_cst = fl.functions.Constant()
    model = fl.models.OverdampedHidden(fun_lin, fun_cst.copy(), fun_cst, dim=1, dim_h=2)
    estimator = fl.EMEstimator(fl.EulerDensity(model), max_iter=5, verbose=3, verbose_interval=1)
    model = estimator.fit_fetch(data_short)
    assert model.fitted_
