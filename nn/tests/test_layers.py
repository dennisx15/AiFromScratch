from nn.backend import xp
import pytest

from nn.layers.activations import ReLU, Sigmoid,Swish

#go to terminal and navigate to tests then run: pytest test_layers.py

def test_relu_forward():
    """
    Function to test the forward pass of the ReLU activation function.
    """
    input_tensor = xp.array([1,-1,2])
    expected_output = xp.array([1,0,2])
    relu = ReLU()
    output = relu.forward(input_tensor)

    assert xp.array_equal(output, expected_output)
    assert output.shape == expected_output.shape


def test_relu_backward():
    """
    Function to test the backward pass of the ReLU activation function.
    """
    grad_output = xp.array([2,2,2])#Gradient with respect to output-> dL/dY
    expected_grad_input = xp.array([2,0,2])
    input_tensor = xp.array([1, -1, 2])
    relu = ReLU()
    relu.forward(input_tensor)

    grad_input = relu.backward(grad_output)#Gradient with respect to input-> dL/dX = dL/dY * dY/dX

    assert xp.array_equal(grad_input, expected_grad_input)
    assert grad_input.shape == input_tensor.shape

def test_sigmoid_forward():
    """
    Function to test the forward pass of the sigmoid activation function.
    """
    sigmoid = Sigmoid()

    X = xp.array([0])
    output = sigmoid.forward(X)

    expected = xp.array([0.5])

    assert xp.allclose(output, expected)
    assert output.shape == X.shape


def test_sigmoid_backward():
    sigmoid = Sigmoid()

    X = xp.array([0])
    sigmoid.forward(X)

    grad_output = xp.array([1])  # simplest case
    grad_input = sigmoid.backward(grad_output)

    expected = xp.array([0.25])  # 0.5 * (1 - 0.5)

    assert xp.allclose(grad_input, expected)


def test_swish_forward():
    swish = Swish()

    X = xp.array([0])
    output = swish.forward(X)

    expected = xp.array([0.0])  # 0 * sigmoid(0)

    assert xp.allclose(output, expected)
    assert output.shape == X.shape


def test_swish_forward_multiple():
    swish = Swish()

    X = xp.array([-1.0, 0.0, 1.0])
    output = swish.forward(X)

    # approximate known values
    expected = xp.array([
        -1 * (1 / (1 + xp.exp(1))),   # ≈ -0.2689
        0.0,
        1 * (1 / (1 + xp.exp(-1)))    # ≈ 0.7310
    ])

    assert xp.allclose(output, expected, atol=1e-3)


def test_swish_backward():
    swish = Swish()

    X = xp.array([0.0])
    swish.forward(X)

    grad_output = xp.array([1.0])
    grad_input = swish.backward(grad_output)

    # sigmoid(0) = 0.5
    expected = xp.array([0.5])  # 0.5 + 0 * ...

    assert xp.allclose(grad_input, expected)