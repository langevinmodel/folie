import pytest
from folie._numpy import np
import torch


@pytest.mark.parametrize("method", ["SGD", "Adam", "RMSprop"])
def test_pytorchminimize(method):
    from folie.utils import pytorch_minimize

    def custom_function(x):
        return torch.sum((x - 3) ** 2) + 2 * x[0]

    # Starting point
    x0 = np.array([0.0, 0.0])

    # Minimization
    result = pytorch_minimize(custom_function, x0, method=method)

    # Assert the result
    expected_result = np.array([2.0, 3.0])
    np.testing.assert_allclose(result, expected_result, rtol=5e-3)
