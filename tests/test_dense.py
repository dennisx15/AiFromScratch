from nn.backend import xp
import pytest
from nn.layers.dense import Dense

def test_dense_forward():
    """
    Function to test the dense layer's forward pass.
    """
    dense = Dense(3, 2)

    dense.W = xp.array([
        [2, 1],
        [1, 1],
        [1, 1]
    ])
    dense.b = xp.array([[0, 0]])

    X = xp.array([[1, 1, 1]])

    output = dense.forward(X)

    expected = xp.array([[4, 3]])

    assert xp.array_equal(output, expected)
    assert output.shape == expected.shape

def test_dense_backward():
    """
    Function to test the dense layer's backward pass.
    """
    dense = Dense(3, 2)

    dense.W = xp.array([
        [2, 1],
        [1, 1],
        [1, 1]
    ])
    dense.b = xp.array([[0, 0]])

    X = xp.array([[1, 1, 1]])
    dense.forward(X)

    grad_output = xp.array([[2, 3]])

    grad_input = dense.backward(grad_output)

    expected_dW = xp.array([
        [2, 3],
        [2, 3],
        [2, 3]
    ])

    expected_db = xp.array([[2, 3]])

    expected_dX = xp.array([[7, 5, 5]])

    assert xp.array_equal(dense.dW, expected_dW)
    assert xp.array_equal(dense.db, expected_db)
    assert xp.array_equal(grad_input, expected_dX)

def test_dense_forward_batch():
    """
    Function to test the dense layer's forward pass with a batched input.
    """
    dense = Dense(3, 2)

    dense.W = xp.array([
        [2, 1],
        [1, 1],
        [1, 1]
    ])
    dense.b = xp.array([[0, 0]])

    X = xp.array([
        [1, 1, 1],
        [2, 2, 2]
    ])#A batch of two inputs

    output = dense.forward(X)

    expected = xp.array([
        [4, 3],
        [8, 6]
    ])

    assert xp.array_equal(output, expected)
    assert output.shape == expected.shape


def test_dense_backward_batch():
    """
    Function to test the dense layer's backward pass with a batched input.
    """
    dense = Dense(3, 2)

    dense.W = xp.array([
        [2, 1],
        [1, 1],
        [1, 1]
    ])
    dense.b = xp.array([[0, 0]])

    X = xp.array([
        [1, 1, 1],
        [2, 2, 2]
    ])

    dense.forward(X)

    grad_output = xp.array([
        [2, 3],
        [1, 1]
    ])

    grad_input = dense.backward(grad_output)

    expected_dW = xp.array([
        [4, 5],
        [4, 5],
        [4, 5]
    ])

    expected_db = xp.array([[3, 4]])

    expected_dX = xp.array([
        [7, 5, 5],
        [3, 2, 2]
    ])

    assert xp.array_equal(dense.dW, expected_dW)
    assert xp.array_equal(dense.db, expected_db)
    assert xp.array_equal(grad_input, expected_dX)