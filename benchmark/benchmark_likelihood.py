import pytest
import os
import numpy as np
import folie as fl
import dask.array as da
import skfem


@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/datasets/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    trj_list.append(trj[:, 1:2])
    trj_list.stats
    return trj_list


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize("transitioncls", [fl.EulerDensity, fl.OzakiDensity, fl.ShojiOzakiDensity, fl.ElerianDensity, fl.KesslerDensity, fl.DrozdovDensity])
def test_likelihood_functions(data, request, benchmark, transitioncls):
    fun = fl.functions.Linear()
    model = fl.models.Overdamped(fun, fun.copy())
    transition = transitioncls(model)
    for i, trj in enumerate(data):
        transition.preprocess_traj(trj)
    transition.use_jac = False
    loglikelihood = benchmark(transition, data.weights[0], data[0], np.array([1.0, 1.0]))
    assert len(loglikelihood) == 1 or len(loglikelihood) == 2


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
@pytest.mark.parametrize(
    "local_fun, parameters",
    [
        (fl.functions.BSplinesFunction, {}),
        (fl.functions.FiniteElement, {"element": skfem.ElementLineP1()}),
    ],
)
def test_local_likelihood(data, request, benchmark, local_fun, parameters):
    n_knots = 20
    epsilon = 1e-10
    domain = fl.MeshedDomain.create_from_range(np.linspace(data.stats.min - epsilon, data.stats.max + epsilon, n_knots))
    fun = local_fun(domain, **parameters)

    model = fl.models.Overdamped(fun)
    transition = fl.EulerDensity(model)
    for i, trj in enumerate(data):
        transition.preprocess_traj(trj)
    # transition.use_jac = False
    loglikelihood = benchmark(transition, data.weights[0], data[0], np.concatenate((np.zeros(n_knots), np.zeros(n_knots) + 1.0)))
    assert len(loglikelihood) == 2


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_hiddenlikelihood_functions(data, request, benchmark):
    fun_lin = fl.functions.Linear()
    fun_cst = fl.functions.Constant().resize((3, 3))
    model = fl.models.OverdampedHidden(fun_lin, fun_lin.copy(), fun_cst, dim=1, dim_h=2)
    transition = fl.EulerDensity(model)

    for i, trj in enumerate(data):
        transition.preprocess_traj(trj)

    mu0 = np.zeros(model.dim_h)
    sig0 = np.identity(model.dim_h)
    e_step = benchmark(transition.e_step, data.weights[0], data[0], model.coefficients, mu0, sig0)
    assert len(e_step) == 2
