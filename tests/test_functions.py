import pytest
import numpy as np
import folie as fl


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.Constant, {}, 1),
        (fl.functions.BSplinesFunction, {"knots": 7}, 7),
    ],
)
def test_functions(fct, parameters, expected):
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fct(**parameters).fit(data).one()

    assert fun(data).shape == (25,)

    assert fun.grad_x(data).shape == (25, 1)

    assert fun.grad_coeffs(data).shape == (25, expected)


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.Constant, {}, 1),
        (fl.functions.BSplinesFunction, {"knots": 7}, 7),
    ],
)
def test_functions_ND(fct, parameters, expected):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun = fct((2,), **parameters).fit(data).one()

    assert fun(data).shape == (12, 2)

    assert fun.grad_x(data).shape == (12, 2, 2)

    assert fun.grad_coeffs(data).shape == (12, 2, expected)


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.Constant, {}, 1),
        (fl.functions.BSplinesFunction, {"knots": 7}, 7),
    ],
)
def test_functions_ND_various_dim(fct, parameters, expected):
    data = np.linspace(-1, 1, 24).reshape(-1, 3)
    fun = fct((4,), **parameters).fit(data).one()

    assert fun(data).shape == (8, 4)

    assert fun.grad_x(data).shape == (8, 4, 2)

    assert fun.grad_coeffs(data).shape == (12, 4, expected)


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.Constant, {}, 1),
    ],
)
def test_matrix_functions_ND(fct, parameters, expected):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun = fct(**parameters).fit(data).reshape((2, 2)).one()

    assert fun(data).shape == (12, 2, 2)

    assert fun.grad_x(data).shape == (12, 2, 2, 2)

    assert fun.grad_coeffs(data).shape == (12, 2, 2, expected)


def test_functions_sum():
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun1 = fl.functions.Constant((1,)).fit(data).zero()

    fun2 = fl.functions.Constant((1,)).fit(data).one()

    fun_sum = fun1 + fun2

    assert fun_sum(data).shape == (25, 1)

    assert fun_sum.grad_x(data).shape == (25, 1, 1)

    assert fun_sum.grad_coeffs(data).shape == (25, 1, 2)


@pytest.mark.skip(reason="Not implemented yet")
def test_functions_tensor():
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun1 = fl.functions.Linear().fit(data).one()

    fun2 = fl.functions.Linear().fit(data).one()

    fun_ten = fun1 * fun2

    assert fun_ten(data).shape == (24, 2)

    assert fun_ten.grad_x(data).shape == (25, 2, 2)

    assert fun_ten.grad_coeffs(data).shape == (25, 2, 2)


@pytest.mark.skip(reason="Change for more complex functions")
def test_functions_composition():
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun1 = fl.functions.Constant((2,)).fit(data).one()

    fun2 = fl.functions.Constant().fit(data).one()

    fun_compo = fl.functions.FunctionComposition(fun1, fun2)

    assert fun_compo(data).shape == (24,)

    assert fun_compo.grad_x(data).shape == (24, 2)

    assert fun_compo.grad_coeffs(data).shape == (24, 2)
