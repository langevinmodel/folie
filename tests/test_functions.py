import pytest
import numpy as np
import folie as fl
import scipy.optimize


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
    ],
)
def test_functions(fct, parameters):
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fct(**parameters).fit(data)

    assert fun(data).shape == (25,)

    assert fun.grad_x(data).shape == (25, 1)

    finite_diff_jac = scipy.optimize.approx_fprime(data[0], lambda x: fun(np.asarray([x]))[0])
    np.testing.assert_allclose(fun.grad_x(data[0:1])[0], finite_diff_jac, rtol=1e-06)

    assert fun.grad_coeffs(data).shape == (25, fun.size)

    def eval_fun(c):
        fun.coefficients = c
        return fun(data[0:1])[0]

    finite_diff_jac = scipy.optimize.approx_fprime(fun.coefficients, eval_fun)
    np.testing.assert_allclose(fun.grad_coeffs(data[0:1])[0], finite_diff_jac, rtol=1e-06)


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
    ],
)
def test_functions_ND(fct, parameters):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun = fct((2,), **parameters).fit(data)

    assert fun(data).shape == (12, 2)

    assert fun.grad_x(data).shape == (12, 2, 2)

    finite_diff_jac = scipy.optimize.approx_fprime(data[0, :], lambda x: fun(x.reshape(1, -1))[0, :])
    np.testing.assert_allclose(fun.grad_x(data[0:1, :])[0, :], finite_diff_jac, rtol=1e-04)

    assert fun.grad_coeffs(data).shape == (12, 2, fun.size)


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
    ],
)
def test_functions_ND_various_dim(fct, parameters):
    data = np.linspace(-1, 1, 24).reshape(-1, 3)
    fun = fct((4,), **parameters).fit(data)

    assert fun(data).shape == (8, 4)

    assert fun.grad_x(data).shape == (8, 4, 3)

    assert fun.grad_coeffs(data).shape == (8, 4, fun.size)


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
    ],
)
def test_matrix_functions_ND(fct, parameters):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun = fct(**parameters).fit(data).resize((2, 2))

    assert fun(data).shape == (12, 2, 2)

    assert fun.grad_x(data).shape == (12, 2, 2, 2)

    assert fun.grad_coeffs(data).shape == (12, 2, 2, fun.size)


def test_functions_sum():
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun1 = fl.functions.Linear((1,)).fit(data)

    fun2 = fl.functions.Linear((1,)).fit(data)

    fun_sum = fun1 + fun2

    assert fun_sum(data).shape == (25, 1)

    assert fun_sum.grad_x(data).shape == (25, 1, 1)

    assert fun_sum.grad_coeffs(data).shape == (25, 1, 2)


@pytest.mark.skip(reason="Not implemented yet")
def test_functions_tensor():
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun1 = fl.functions.Linear().fit(data)

    fun2 = fl.functions.Linear().fit(data)

    fun_ten = fun1 * fun2

    assert fun_ten(data).shape == (24, 2)

    assert fun_ten.grad_x(data).shape == (25, 2, 2)

    assert fun_ten.grad_coeffs(data).shape == (25, 2, 2)


@pytest.mark.skip(reason="Change for more complex functions")
def test_functions_composition():
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun1 = fl.functions.Linear((2,)).fit(data)

    fun2 = fl.functions.Linear().fit(data)

    fun_compo = fl.functions.FunctionComposition(fun1, fun2)

    assert fun_compo(data).shape == (24,)

    assert fun_compo.grad_x(data).shape == (24, 2)

    assert fun_compo.grad_coeffs(data).shape == (24, 2)
