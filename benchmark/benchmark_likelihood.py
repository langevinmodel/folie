import pytest
import os
import numpy as np
import folie as fl
import dask.array as da


@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    trj_list.append(trj[:, 1:2])
    trj_list.stats
    return trj_list


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("transitioncls", [fl.EulerDensity, fl.OzakiDensity, fl.ShojiOzakiDensity, fl.ElerianDensity, fl.KesslerDensity, fl.DrozdovDensity])
def test_likelihood_bf(data, request, benchmark, transitioncls):
    bf = fl.function_basis.Linear().fit(data)
    model = fl.models.OverdampedBF(bf)
    transition = transitioncls(model)
    loglikelihood = benchmark(transition, data.weights[0], data[0], np.array([1.0, 1.0]))
    assert len(loglikelihood) == 1


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("transitioncls", [fl.EulerDensity, fl.OzakiDensity, fl.ShojiOzakiDensity, fl.ElerianDensity, fl.KesslerDensity, fl.DrozdovDensity])
def test_likelihood_functions(data, request, benchmark, transitioncls):
    fun = fl.functions.Linear().fit(data)
    model = fl.models.OverdampedFunctions(fun, fun.copy())
    transition = transitioncls(model)
    loglikelihood = benchmark(transition, data.weights[0], data[0], np.array([1.0, 1.0]))
    assert len(loglikelihood) == 1


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
def test_numba_optimized(data, request, benchmark):
    n_knots = 20
    epsilon = 1e-10
    model = fl.models.OverdampedFreeEnergy(np.linspace(data.stats.min - epsilon, data.stats.max + epsilon, n_knots), 1.0)
    transition = fl.EulerNumbaOptimizedDensity(model)
    for i, trj in enumerate(data):
        transition.preprocess_traj(trj)
    loglikelihood = benchmark(transition, data.weights[0], data[0], np.concatenate((np.zeros(n_knots), np.zeros(n_knots) + 1.0)))
    assert len(loglikelihood) == 2
