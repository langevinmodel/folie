import pytest
from folie._numpy import np
import folie as fl
import scipy.optimize
from sklearn.kernel_ridge import KernelRidge
import skfem


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.Polynomial, {"deg": 3}),
        (fl.functions.Polynomial, {"deg": 3, "polynom": np.polynomial.Chebyshev}),
        (fl.functions.BSplinesFunction, {}),
        (fl.functions.Fourier, {"order": 3}),
        (fl.functions.RadialBasisFunction, {}),
    ],
)
def test_functions(fct, parameters):
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    domain = fl.MeshedDomain.create_from_range(np.linspace(-1, 1, 7))
    fun = fct(domain=domain, **parameters)

    assert fun(data).shape == (25,)

    assert fun.grad_x(data).shape == (25, 1)

    finite_diff_jac = scipy.optimize.approx_fprime(data[0], lambda x: fun(np.asarray([x]))[0])
    np.testing.assert_allclose(fun.grad_x(data[0:1])[0], finite_diff_jac, rtol=1e-06)

    assert fun.grad_coeffs(data).shape == (25, fun.size)

    def eval_fun(c):
        fun.coefficients = c
        return fun(data[0:1])[0]

    finite_diff_jac = scipy.optimize.approx_fprime(fun.coefficients, eval_fun)
    np.testing.assert_allclose(fun.grad_coeffs(data[0:1])[0], finite_diff_jac, atol=1e-6, rtol=1e-6)


def test_fem_functions():

    m = skfem.MeshTri().refined(4)
    domain = fl.MeshedDomain(m)
    e = skfem.ElementTriP1()
    fun = fl.functions.FiniteElement(domain, e)
    data = np.linspace(0, 1, 24).reshape(-1, 2)
    fun.fit(data, np.ones(12))
    assert fun(data).shape == (12,)

    assert fun.grad_x(data).shape == (12, 2)

    finite_diff_jac = scipy.optimize.approx_fprime(data[0], lambda x: fun(np.asarray([x]))[0])
    np.testing.assert_allclose(fun.grad_x(data[0:1])[0], finite_diff_jac, rtol=1e-06)

    assert fun.grad_coeffs(data).shape == (12, fun.size)

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
        (fl.functions.Polynomial, {"deg": 3}),
        (fl.functions.Polynomial, {"deg": 3, "polynom": np.polynomial.Chebyshev}),
        (fl.functions.Fourier, {"order": 3}),
        (fl.functions.RadialBasisFunction, {}),
    ],
)
def test_functions_ND(fct, parameters):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    m = skfem.MeshTri().refined(4)
    fun = fct(domain=fl.MeshedDomain(m), output_shape=(2,), **parameters).fit(data)

    assert fun(data).shape == (12, 2)

    assert fun.grad_x(data).shape == (12, 2, 2)

    finite_diff_jac = scipy.optimize.approx_fprime(data[0, :], lambda x: fun(x.reshape(1, -1))[0, :])
    np.testing.assert_allclose(fun.grad_x(data[0:1, :])[0, :], finite_diff_jac, rtol=1e-06)

    assert fun.grad_coeffs(data).shape == (12, 2, fun.size)


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.Polynomial, {"deg": 3}),
        (fl.functions.Polynomial, {"deg": 3, "polynom": np.polynomial.Chebyshev}),
        (fl.functions.Fourier, {"order": 3}),
    ],
)
def test_functions_ND_various_dim(fct, parameters):
    data = np.linspace(-1, 1, 24).reshape(-1, 3)
    fun = fct(domain=fl.Domain.Rd(3), output_shape=(4,), **parameters).fit(data)

    assert fun(data).shape == (8, 4)

    assert fun.grad_x(data).shape == (8, 4, 3)

    assert fun.grad_coeffs(data).shape == (8, 4, fun.size)


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.Constant, {}),
        (fl.functions.Linear, {}),
        (fl.functions.Polynomial, {"deg": 3}),
        (fl.functions.Polynomial, {"deg": 3, "polynom": np.polynomial.Chebyshev}),
        (fl.functions.Fourier, {"order": 3}),
        (fl.functions.RadialBasisFunction, {}),
    ],
)
def test_matrix_functions_ND(fct, parameters):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    m = skfem.MeshTri().refined(4)
    fun = fct(domain=fl.MeshedDomain(m), **parameters).fit(data).resize((2, 2))

    assert fun(data).shape == (12, 2, 2)

    assert fun.grad_x(data).shape == (12, 2, 2, 2)

    assert fun.grad_coeffs(data).shape == (12, 2, 2, fun.size)


def test_functions_sum():
    data = np.linspace(-1, 1, 25).reshape(-1, 1)

    fun1 = fl.functions.Linear(output_shape=(1,)).fit(data)

    fun2 = fl.functions.Polynomial(deg=3, output_shape=(1,)).fit(data)

    fun_sum = fun1 + fun2

    assert fun_sum(data).shape == (25, 1)

    assert fun_sum.grad_x(data).shape == (25, 1, 1)

    assert fun_sum.grad_coeffs(data).shape == (25, 1, 5)


@pytest.mark.skip(reason="Not implemented yet")
def test_functions_tensor():
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    fun1 = fl.functions.Linear().fit(data)

    fun2 = fl.functions.Linear().fit(data)

    fun_ten = fun1 * fun2

    assert fun_ten(data).shape == (24, 2)

    assert fun_ten.grad_x(data).shape == (25, 2, 2)

    assert fun_ten.grad_coeffs(data).shape == (25, 2, 2)


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.KernelFunction, {"gamma": 0.5}),
        (fl.functions.sklearnWrapper, {"estimator": KernelRidge()}),
    ],
)
def test_nonparametricfunctions(fct, parameters):
    data = np.linspace(-1, 1, 25).reshape(-1, 1)
    y = data**2
    fun = fct(domain=fl.Domain.Rd(1), **parameters).fit(data, y)

    assert fun(data).shape == (25,)

    assert fun.grad_x(data).shape == (25, 1)

    finite_diff_jac = scipy.optimize.approx_fprime(data[0], lambda x: fun(np.asarray([x]))[0])
    np.testing.assert_allclose(fun.grad_x(data[0:1])[0], finite_diff_jac, rtol=1e-06)


@pytest.mark.parametrize(
    "fct,parameters",
    [
        (fl.functions.KernelFunction, {"gamma": 0.5}),
        (fl.functions.sklearnWrapper, {"estimator": KernelRidge()}),
    ],
)
def test_nonparametricfunctions_ND(fct, parameters):
    data = np.linspace(-1, 1, 24).reshape(-1, 2)
    y = data[..., None] * data[:, None, :]

    fun = fct(domain=fl.Domain.Rd(2), output_shape=(2, 2), **parameters).fit(data, y)

    assert fun(data).shape == (12, 2, 2)

    assert fun.grad_x(data).shape == (12, 2, 2, 2)

    finite_diff_jac = scipy.optimize.approx_fprime(data[0, :], lambda x: fun(x.reshape(1, -1)).ravel())
    np.testing.assert_allclose(fun.grad_x(data[0:1, :])[0, :], finite_diff_jac.reshape(2, 2, 2), rtol=1e-06)


def test_numerical_difference():
    X = np.linspace(-1, 1, 24).reshape(-1, 2)
    domain = fl.Domain.Rd(2)
    fun = fl.functions.Fourier(order=3, domain=domain).fit(X, (X**2).sum(axis=1))

    finite_diff = fl.functions.approx_fprime(X, fun, fun.output_shape_)

    finite_diff_jac = np.zeros((X.shape[0], X.shape[1]))
    for n in range(X.shape[0]):
        finite_diff_jac[n, :] = scipy.optimize.approx_fprime(X[n, :], lambda x: fun(x.reshape(1, -1)))
    np.testing.assert_allclose(finite_diff, finite_diff_jac, rtol=1e-06)


@pytest.mark.parametrize(
    "pot",
    [
        fl.functions.ConstantForce(),
        fl.functions.Quadratic(),
        fl.functions.Quartic(),
        fl.functions.Cosine(),
        fl.functions.Quartic2D(),
        fl.functions.EntropicSwitch(),
        fl.functions.MullerBrown(),
        fl.functions.RuggedMullerBrown(),
        fl.functions.LogExpPot(),
        fl.functions.ValleyRidgePotential(),
        fl.functions.SimpleValleyRidgePotential(),
    ],
)
def test_analytical_potential(pot):

    X = np.linspace(-1, 1, 25 * pot.dim).reshape(-1, pot.dim)

    assert pot.size == len(pot.coefficients)

    assert pot.potential(X).shape == (25,)

    assert pot.force(X).shape == (25, pot.dim)

    if pot.dim == 1:
        assert pot(X).shape == (25,)
    else:
        assert pot(X).shape == (25, pot.dim)
