import pytest
from folie._numpy import np
import folie as fl


@pytest.mark.parametrize("steppercls", [fl.simulations.EulerStepper, fl.simulations.MilsteinStepper])
def test_simple_simulation(steppercls):

    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fl.functions.Polynomial(deg=3).fit(data, data[:, 0])
    model = fl.models.Overdamped(fun)

    simu_engine = fl.Simulator(steppercls(model), 1e-3)

    trj_data = simu_engine.run(50, [0.0])

    assert len(trj_data) == 1

    assert trj_data[0]["x"].shape == (50, 1)


@pytest.mark.parametrize("steppercls", [fl.simulations.VECStepper])
def test_underdamped_simulation(steppercls):

    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fl.functions.Polynomial(deg=3).fit(data, data[:, 0])
    model = fl.models.Underdamped(fun, fun, fun)

    simu_engine = fl.simulations.UnderdampedSimulator(steppercls(model), 1e-3)

    trj_data = simu_engine.run(50, [0.0, 0.1])

    assert len(trj_data) == 1

    assert trj_data[0]["x"].shape == (50, 1)

    assert trj_data[0]["v"].shape == (50, 1)


@pytest.mark.parametrize("model", [fl.models.BrownianMotion(), fl.models.OrnsteinUhlenbeck()])  # , fl.models.BrownianMotion(dim=3), fl.models.OrnsteinUhlenbeck(dim=3)])
def test_exact_simulation(model):
    simu_engine = fl.Simulator(fl.simulations.ExactStepper(model), 1e-3)

    trj_data = simu_engine.run(50, [0.0])

    assert len(trj_data) == 1

    assert trj_data[0]["x"].shape == (50, 1)

    # assert length of trj


def test_abmd_simulation():
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fl.functions.Polynomial(deg=3).fit(data, data[:, 0])
    model = fl.models.Overdamped(fun)

    simu_engine = fl.simulations.ABMD_Simulator(fl.simulations.EulerStepper(model), 1e-3, k=2.0, xstop=2.0)

    trj_data = simu_engine.run(50, [0.0])

    # assert length of traj et présence de biais ddedans

    assert len(trj_data) == 1

    assert trj_data[0]["x"].shape == (50, 1)

    assert len(trj_data) == 1

    assert trj_data[0]["bias"].shape == (50, 1)
