from nn.backend import xp
from nn.losses.mse import MSE


def test_mse_forward():
    """
    Function to test MSE forward pass
    """
    mse = MSE()

    y_pred = xp.array([2.0])
    y_true = xp.array([1.0])

    loss = mse.forward(y_pred, y_true)

    expected = 1.0  # (2 - 1)^2

    assert xp.isclose(loss, expected)

def test_mse_backward():
    """
    Function to test MSE backward pass
    """
    mse = MSE()

    y_pred = xp.array([2.0])
    y_true = xp.array([1.0])

    mse.forward(y_pred, y_true)
    grad = mse.backward()

    expected = xp.array([2.0])  # 2*(2 - 1)

    assert xp.allclose(grad, expected)


def test_mse_batch():
    """
    Function to test MSE forward and backward pass with a batch
    """
    mse = MSE()

    y_pred = xp.array([2.0, 3.0])
    y_true = xp.array([1.0, 1.0])

    loss = mse.forward(y_pred, y_true)

    expected_loss = ((1**2 + 2**2) / 2)  # mean

    assert xp.isclose(loss, expected_loss)

    grad = mse.backward()

    expected_grad = xp.array([2*(2-1)/2, 2*(3-1)/2])

    assert xp.allclose(grad, expected_grad)