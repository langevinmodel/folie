import pytest
import os
import numpy as np
import folie as fl
import dask.array as da
import scipy.optimize


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
@pytest.mark.parametrize("transitioncls", [fl.OzakiDensity, fl.ShojiOzakiDensity, fl.ElerianDensity, fl.KesslerDensity, fl.DrozdovDensity])
def test_likelihood_bf(data, request, transitioncls):
    bf = fl.function_basis.Linear().fit(data)
    model = fl.models.OverdampedBF(bf)
    transition = transitioncls(model)
    transition.preprocess_traj(data[0])
    loglikelihood = transition(data.weights[0], data[0], np.array([1.0, 1.0]))
    assert len(loglikelihood) == 1


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "transitioncls",
    [
        fl.EulerDensity,
    ],
)
def testlikelihood_derivative(data, request, transitioncls):
    fun_lin = fl.functions.Linear().fit(data)
    model = fl.models.OverdampedFunctions(fun_lin)
    transition = transitioncls(model)
    transition.preprocess_traj(data[0])
    loglikelihood = transition(data.weights[0], data[0], np.array([1.0, 1.0]))
    assert len(loglikelihood) == 2

    assert loglikelihood[1].shape == (len(model.coefficients),)
    # Testing for evaluation of the jacobian
    finite_diff_jac = scipy.optimize.approx_fprime(model.coefficients, lambda p: transition(data.weights[0], data[0], p)[0])
    np.testing.assert_allclose(loglikelihood[1], finite_diff_jac, rtol=1e-05)


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
@pytest.mark.parametrize(
    "transitioncls",
    [
        fl.EulerHiddenDensity,
    ],
)
def testlikelihoodND_derivative(data, request, transitioncls):
    fun_lin = fl.functions.Linear().fit(data)
    model = fl.models.OverdampedFunctions(fun_lin, dim=1)
    transition = transitioncls(model)
    transition.preprocess_traj(data[0])
    loglikelihood = transition(data.weights[0], data[0], np.array([1.0, 1.0]))
    assert len(loglikelihood) == 2

    assert loglikelihood[1].shape == (len(model.coefficients),)
    # Testing for evaluation of the jacobian
    finite_diff_jac = scipy.optimize.approx_fprime(model.coefficients, lambda p: transition(data.weights[0], data[0], p)[0])
    np.testing.assert_allclose(loglikelihood[1], finite_diff_jac, rtol=1e-06, atol=1e-6)


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
@pytest.mark.parametrize(
    "transitioncls",
    [
        fl.EulerHiddenDensity,
    ],
)
@pytest.mark.parametrize("dim_h", [1])
def testcorrection_hiddenND_derivative(data, request, transitioncls, dim_h):
    fun_lin = fl.functions.Linear().fit(data)
    fun_cst = fl.functions.Constant().fit(data)
    model = fl.models.OverdampedHidden(fun_lin, fun_cst.copy(), fun_cst, dim=1, dim_h=dim_h)
    A = np.block([[1, -0.5 * np.ones(dim_h)], [-0.7 * np.ones(dim_h), 2 * np.eye(dim_h)]])
    model.coefficients_diffusion = A @ A.T
    transition = transitioncls(model)
    transition.preprocess_traj(data[0])

    muh0, sigh0 = transition.e_step(data.weights[0], data[0], model.coefficients, np.zeros(dim_h), np.eye(dim_h))
    assert muh0.shape == (dim_h,)
    assert sigh0.shape == (dim_h, dim_h)

    correction = transition.hiddencorrection(data.weights[0], data[0], model.coefficients)
    assert len(correction) == 2

    assert correction[1].shape == (len(model.coefficients),)
    # Testing for evaluation of the jacobian
    finite_diff_jac = scipy.optimize.approx_fprime(model.coefficients, lambda p: transition.hiddencorrection(data.weights[0], data[0], p)[0])
    np.testing.assert_allclose(correction[1], finite_diff_jac, rtol=1e-06, atol=1e-6)


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
def test_numba_optimized(data, request):
    n_knots = 20
    epsilon = 1e-10
    model = fl.models.OverdampedFreeEnergy(np.linspace(data.stats.min - epsilon, data.stats.max + epsilon, n_knots), 1.0)
    transition = fl.EulerNumbaOptimizedDensity(model)
    for i, trj in enumerate(data):
        transition.preprocess_traj(trj)
    # Assert preprocessing as well
    coeffs0 = np.concatenate((np.zeros(n_knots), np.zeros(n_knots) + 1.0))
    loglikelihood = transition(data.weights[0], data[0], coeffs0)
    assert len(loglikelihood) == 2

    assert loglikelihood[1].shape == (2 * n_knots,)

    # Testing for evaluation of the jacobian
    jac = scipy.optimize.approx_fprime(coeffs0, lambda p: transition(data.weights[0], data[0], p)[0], 1e-9)
    np.testing.assert_allclose(loglikelihood[1], jac, rtol=1e-06, atol=1e-6)
