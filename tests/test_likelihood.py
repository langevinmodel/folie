import pytest
import os
from folie._numpy import np, value_and_grad
import folie as fl
import torch
import scipy.optimize
from numpy.testing import assert_allclose

@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/datasets/example_2d.trj"))
    if request.param == "torch":
        trj = torch.from_numpy(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    trj_list.append(trj[:, 1:2])
    trj_list.stats
    return trj_list


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
@pytest.mark.parametrize("transitioncls", [fl.EulerDensity,fl.ElerianDensity, fl.KesslerDensity, fl.DrozdovDensity])
def test_likelihood(data, request, transitioncls):
    # print(data.representative_array())
    fun_lin = fl.functions.Linear().fit(data.representative_array(), np.ones(data.representative_array().shape[0]))
    model = fl.models.Overdamped(fun_lin, dim=1)
    transition = transitioncls(model)
    transition.preprocess_traj(data[0])

    loglikelihood, jac = value_and_grad(lambda p: transition(data.weights[0], data[0], p))(np.array([1.0, 1.0]))

    # Testing for evaluation of the jacobian
    finite_diff_jac = scipy.optimize.approx_fprime(model.coefficients._value, lambda p: transition(data.weights[0], data[0], p))
    assert_allclose(jac, finite_diff_jac, rtol=1e-06, atol=1e-6)
    



@pytest.mark.parametrize("data", ["numpy"], indirect=True)
@pytest.mark.parametrize(
    "transitioncls",
    [
        fl.EulerDensity,
    ],
)
def testlikelihoodND_derivative(data, request, transitioncls):
    domain = fl.MeshedDomain.create_from_range(np.linspace(data.stats.min, data.stats.max, 7))
    fun_lin = fl.functions.BSplinesFunction(domain).fit(data.representative_array(), np.ones(data.representative_array().shape[0]))
    model = fl.models.Overdamped(fun_lin)
    transition = transitioncls(model)
    transition.preprocess_traj(data[0])
    loglikelihood = value_and_grad(lambda p: transition(data.weights[0], data[0], p))(model.coefficients)

    # Testing for evaluation of the jacobian
    finite_diff_jac = scipy.optimize.approx_fprime(model.coefficients._value, lambda p: transition(data.weights[0], data[0], p))
    assert_allclose(loglikelihood[1], finite_diff_jac, rtol=1e-06, atol=1e-6)


@pytest.mark.parametrize("data", ["numpy"], indirect=True)
@pytest.mark.parametrize(
    "transitioncls",
    [
        fl.EulerDensity,
    ],
)
@pytest.mark.parametrize("dim_h", [1, 2])
def testcorrection_hiddenND_derivative(data, request, transitioncls, dim_h):
    fun_lin = fl.functions.Linear().fit(data.representative_array(), np.ones(data.representative_array().shape[0]))
    fun_cst = fl.functions.Constant().fit(data.representative_array(), np.ones(data.representative_array().shape[0]))
    model = fl.models.OverdampedHidden(fun_lin, fun_cst.copy(), fun_cst, dim=1, dim_h=dim_h)
    A = np.block([[np.eye(1), -0.5 * np.ones(dim_h)], [-0.7 * np.ones(dim_h).reshape(-1, 1), 2 * np.eye(dim_h)]])
    model.diffusion.coefficients = A @ A.T
    transition = transitioncls(model)
    transition.preprocess_traj(data[0])

    muh0, sigh0 = transition.e_step(data.weights[0], data[0], model.coefficients, np.zeros(dim_h), np.eye(dim_h))
    assert muh0.shape == (dim_h,)
    assert sigh0.shape == (dim_h, dim_h)

    correction = value_and_grad(lambda p: transition.hiddencorrection(data.weights[0], data[0], p))(model.coefficients)

    # TODO: assert also the plain likelihood part of the transitionDensity


    # Testing for evaluation of the jacobian
    finite_diff_jac = scipy.optimize.approx_fprime(model.coefficients._value, lambda p: transition.hiddencorrection(data.weights[0], data[0], p))
    assert_allclose(correction[1], finite_diff_jac, rtol=1e-06, atol=1e-6)

    # TODO: assert also the plain likelihood part of the transitionDensity
