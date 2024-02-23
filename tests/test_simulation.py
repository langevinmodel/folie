import pytest
import numpy as np
import folie as fl


def test_simple_simulation():

    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fl.functions.Polynomial(deg=3).fit(data, data[:, 0])
    model = fl.models.Overdamped(fun)

    simu_engine = fl.Simulator(fl.EulerDensity(model), 1e-3)

    trj_data = simu_engine.run(50, [0.0])

    assert len(trj_data) == 1

    assert trj_data[0]["x"].shape == (50, 1)

    # assert length of trj


@pytest.mark.skip(reason="Not yet implemented")
def test_abmd_simulation():
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    fun = fl.functions.Polynomial(deg=3).fit(data, data[:, 0])
    model = fl.models.Overdamped(fun)

    simu_engine = fl.ABMD_Simulator(fl.EulerDensity(model), 1e-3, k=2.0, xstop=2.0)

    trj_data = simu_engine.run(50, [0.0])

    # assert length of traj et présence de biais ddedans

    assert len(trj_data) == 1

    assert trj_data[0]["x"].shape == (50, 1)

    assert len(trj_data) == 1

    assert trj_data[0]["bias"].shape == (50, 1)
