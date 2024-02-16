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
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
    ],
)
def test_functions(data, request, benchmark, fct, parameters):
    fun = fct(**parameters).fit(data)
    fun_vals = benchmark(fun, data)
    assert fun_vals.shape == (data.shape[0],)


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
    ],
)
def test_functions_gradx(data, request, benchmark, fct, parameters):
    fun = fct(**parameters).fit(data)
    fun_vals = benchmark(fun.grad_x, data)

    assert fun_vals.shape == (data.shape[0], 1)


@pytest.mark.parametrize("data", ["numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.BSplinesFunction, {"knots": 7}),
    ],
)
def test_functions_grad_coeffs(data, request, benchmark, fct, parameters):
    fun = fct(**parameters).fit(data)
    fun_vals = benchmark(fun.grad_coeffs, data)

    assert fun_vals.shape == (data.shape[0], fun.size)
