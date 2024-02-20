import pytest
import os
import numpy as np
import folie as fl
import dask.array as da
import torch


@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    elif request.param == "torch":
        trj = torch.from_numpy(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    for i in range(1, trj.shape[1]):
        trj_list.append(trj[:, i : (i + 1)])
    trj_list.stats
    return trj_list


@pytest.fixture
def data2d(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    elif request.param == "torch":
        trj = torch.from_numpy(trj)
    trj_list = fl.Trajectories(dt=trj[1, 0] - trj[0, 0])
    trj_list.append(trj[:, 1:3])
    trj_list.stats
    return trj_list


@pytest.mark.parametrize("data", ["numpy", "dask", "torch"], indirect=True)
def test_statistics(data, request):
    stats = data.stats

    assert stats.nobs == 600000

    assert data.domain(75).shape == (75, 1)

    rep_array = data.representative_array(75)

    assert rep_array.shape == (75, 1)

    np.testing.assert_allclose(stats.mean, rep_array.mean(), atol=1.0 / rep_array.shape[0])

    np.testing.assert_allclose(stats.variance, rep_array.var(), atol=1 / rep_array.shape[0])

    np.testing.assert_allclose(stats.min, rep_array.min(axis=0))

    np.testing.assert_allclose(stats.max, rep_array.max(axis=0))
