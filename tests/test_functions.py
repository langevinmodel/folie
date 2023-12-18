import pytest
import numpy as np
import folie as fl


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.ConstantFunction, {}, 1),
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
        (fl.functions.ConstantFunction, {}, 1),
        (fl.functions.BSplinesFunction, {"knots": 7}, 7),
    ],
)
def test_functions_ND(fct, parameters, expected):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun = fct((2,), **parameters).fit(data).one()

    assert fun(data).shape == (12, 2)

    assert fun.grad_x(data).shape == (12, 2)

    assert fun.grad_coeffs(data).shape == (12, 2, expected)


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.ConstantFunction, {}, 1),
    ],
)
def test_matrix_functions_ND(fct, parameters, expected):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun = fct("matrix", **parameters).fit(data).one()

    assert fun(data).shape == (12, 2, 2)

    assert fun.grad_x(data).shape == (12, 2, 2)

    assert fun.grad_coeffs(data).shape == (12, 2, 2, expected)


def test_functions_sum():
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun1 = fl.functions.ConstantFunction(1).fit(data).zero()

    fun2 = fl.functions.ConstantFunction(1).fit(data).one()

    fun_sum = fun1 + fun2

    assert fun_sum(data).shape == (25, 1)

    assert fun_sum.grad_x(data).shape == (25, 1)

    assert fun_sum.grad_coeffs(data).shape == (25, 1, 1)


@pytest.mark.skip(reason="Not implemented yet")
def test_functions_tensor():
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun1 = fl.functions.Linear().fit(data).one()

    fun2 = fl.functions.Linear().fit(data).one()

    fun_ten = fun1 * fun2

    assert fun_ten(data).shape == 1

    assert fun_ten.grad_x(data).shape == 1

    assert fun_ten.grad_coeffs(data).shape == 1


@pytest.mark.skip(reason="Change for more complex functions")
def test_functions_composition():
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun1 = fl.functions.ConstantFunction((2,)).fit(data).one()

    fun2 = fl.functions.ConstantFunction().fit(data).one()

    fun_compo = fl.functions.FunctionComposition(fun1, fun2)

    assert fun_compo(data).shape == (24,)

    assert fun_compo.grad_x(data).shape == (24, 2)

    assert fun_compo.grad_coeffs(data).shape == (24, 2)
