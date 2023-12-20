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
def test_overdamped(fct, parameters, expected):
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fct(**parameters).fit(data).one()
    model = fl.models.OverdampedFunctions(fun, fun.copy())

    # model.coefficients =  # Find set of coefficients
    assert model.is_linear
    x = np.linspace(-1, 1, 15)

    assert model.force(x).shape == (15,)

    assert model.diffusion(x).shape == (15,)
