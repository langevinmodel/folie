import pytest
import os
import numpy as np
import folie as fl
import dask.array as da


@pytest.fixture
def data(request):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    trj = np.loadtxt(os.path.join(file_dir, "../examples/example_2d.trj"))
    if request.param == "dask":
        trj = da.from_array(trj)
    return trj[:, 1:2]


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.function_basis.Linear, {}),
        (fl.function_basis.BSplines, {"n_knots": 7}),
    ],
)
def test_functions(data, request, benchmark, fct, parameters):
    bf = fct(**parameters).fit(data)
    basis_vals = benchmark(bf, data)
    assert basis_vals.shape == (data.shape[0], bf.size)
