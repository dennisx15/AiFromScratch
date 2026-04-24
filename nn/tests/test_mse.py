import numpy as np
from nn.losses.mse import MSE


def test_mse_forward():
    mse = MSE()

    y_pred = np.array([2.0])
    y_true = np.array([1.0])

    loss = mse.forward(y_pred, y_true)

    expected = 1.0  # (2 - 1)^2

    assert np.isclose(loss, expected)

def test_mse_backward():

    mse = MSE()

    y_pred = np.array([2.0])
    y_true = np.array([1.0])

    mse.forward(y_pred, y_true)
    grad = mse.backward()

    expected = np.array([2.0])  # 2*(2 - 1)

    assert np.allclose(grad, expected)


def test_mse_batch():
    mse = MSE()

    y_pred = np.array([2.0, 3.0])
    y_true = np.array([1.0, 1.0])

    loss = mse.forward(y_pred, y_true)

    expected_loss = ((1**2 + 2**2) / 2)  # mean

    assert np.isclose(loss, expected_loss)

    grad = mse.backward()

    expected_grad = np.array([2*(2-1)/2, 2*(3-1)/2])

    assert np.allclose(grad, expected_grad)