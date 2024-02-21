import torch


def pytorch_minimize(func, x0, method="Adam", max_iter=1000, tol=1e-6, lr=0.01, **kwargs):
    """
    Interface for using PyTorch optimizers similar to scipy.optimize.minimize.

    Args:
        func (callable): The function to minimize.
        x0 (torch.Tensor): The initial value of the parameters.
        method (str): The name of the PyTorch optimizer to use (default: 'adam').
        max_iter (int): The maximum number of iterations (default: 1000).
        tol (float): The tolerance for convergence (default: 1e-6).
        lr (float): The learning rate (default: 0.01).
        **kwargs: Other arguments specific to the PyTorch optimizer.

    Returns:
        torch.Tensor: The value of the parameters that minimizes the function.
    """
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float64)

    optimizer_cls = getattr(torch.optim, method, None)
    if optimizer_cls is None:
        raise ValueError(f"Unrecognized optimization method: {method}")

    optimizer = optimizer_cls([x], lr=lr, **kwargs)

    prev_loss = float("inf")
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = func(x)
        loss.backward()
        optimizer.step()
        if abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()

    return x.detach().numpy()
