import pytest
from folie._numpy import np
import folie as fl


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.Constant, {}, 1),
        (fl.functions.Polynomial, {"deg": 3}, 4),
        (fl.functions.BSplinesFunction, {"domain": fl.MeshedDomain.create_from_range(np.linspace(-1, 1, 7))}, 7),
    ],
)
def test_overdamped(fct, parameters, expected):
    fun = fct(**parameters)
    model = fl.models.Overdamped(fun, fun.copy())

    x = np.linspace(-1, 1, 15).reshape(-1, 1)
    assert model.pos_drift(x).shape == (15,)

    assert model.diffusion(x).shape == (15,)


@pytest.mark.parametrize(
    "model",
    [fl.models.BrownianMotion(), fl.models.OrnsteinUhlenbeck()],
)
def test_overdamped_w_exactdensity(model):
    x = np.linspace(-1, 1, 15).reshape(-1, 1)

    assert model.pos_drift(x).shape == (15,)

    assert model.diffusion(x).shape == (15,)

    assert model.exact_density(x[1:], x[:-1], 0.0, 1e-3).shape == (14,)

    assert (model.exact_step(np.zeros((1, 1)), 1e-3, 0.1) != 0.0).any()


@pytest.mark.parametrize(
    "model",
    [fl.models.BrownianMotion(dim=3), fl.models.OrnsteinUhlenbeck(dim=3)],
)
def test_overdamped_w_exactdensityND(model):
    x = np.linspace(-1, 1, 45).reshape(-1, 3)

    assert model.pos_drift(x).shape == (15, 3)

    assert model.diffusion(x).shape == (15, 3, 3)

    # assert model.exact_density(x[1:], x[:-1], 0.0, 1e-3).shape == (14,)
    #
    # assert (model.exact_step(np.zeros((1, 1)), 1e-3, 0.1) != 0.0).any()


@pytest.mark.parametrize("model", [fl.models.OverdampedSplines1D(fl.MeshedDomain1D.create_from_range(np.linspace(-1, 1, 7)))])
def test_overdamped_various(model):
    x = np.linspace(-1, 1, 15).reshape(-1, 1)

    assert model.pos_drift(x).shape == (15,)

    assert model.diffusion(x).shape == (15,)


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.Constant, {}, 1),
        (fl.functions.Polynomial, {"deg": 3}, 4),
        (fl.functions.sklearnBSplines, {"domain": fl.MeshedDomain1D.create_from_range(np.linspace(-1, 1, 7))}, 7),
    ],
)
def test_overdamped_ND(fct, parameters, expected):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun = fct(**parameters).fit(data)
    model = fl.models.Overdamped(fun, fun.copy(), dim=2)

    x = np.linspace(-1, 1, 14).reshape(-1, 2)
    assert model.pos_drift(x).shape == (7, 2)

    assert model.diffusion(x).shape == (7, 2, 2)


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.Constant, {}, 1),
        (fl.functions.Polynomial, {"deg": 3}, 4),
        (fl.functions.BSplinesFunction, {"domain": fl.MeshedDomain1D.create_from_range(np.linspace(-1, 1, 7))}, 7),
    ],
)
def test_underdamped(fct, parameters, expected):
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fct(**parameters).fit(data)
    model = fl.models.Underdamped(fun, fun.copy(), fun.copy())

    x = np.linspace(-1, 1, 15).reshape(-1, 1)

    v = np.linspace(-1, 1, 15).reshape(-1, 1)

    assert model.drift(x, v).shape == (15,)

    assert model.pos_drift(x).shape == (15,)

    assert model.friction(x).shape == (15,)

    assert model.diffusion(x).shape == (15,)


@pytest.mark.parametrize(
    "fct,parameters,expected",
    [
        (fl.functions.Constant, {}, 1),
        (fl.functions.Polynomial, {"deg": 3}, 4),
        (fl.functions.sklearnBSplines, {"domain": fl.MeshedDomain1D.create_from_range(np.linspace(-1, 1, 7))}, 7),
    ],
)
def test_underdamped_ND(fct, parameters, expected):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun = fct(**parameters).fit(data)
    model = fl.models.Underdamped(fun, fun.copy(), fun.copy(), dim=2)

    x = np.linspace(-1, 1, 14).reshape(-1, 2)
    v = np.linspace(-1, 1, 14).reshape(-1, 2)

    assert model.drift(x, v).shape == (7, 2)

    assert model.pos_drift(x).shape == (7, 2)

    assert model.friction(x).shape == (7, 2, 2)

    assert model.diffusion(x).shape == (7, 2, 2)
