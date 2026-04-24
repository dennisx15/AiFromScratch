import numpy as np
import pytest
from nn.layers.dense import Dense

def test_dense_forward():
    """
    Function to test the dense layer's forward pass.
    """
    dense = Dense(3, 2)

    dense.W = np.array([
        [2, 1],
        [1, 1],
        [1, 1]
    ])
    dense.b = np.array([[0, 0]])

    X = np.array([[1, 1, 1]])

    output = dense.forward(X)

    expected = np.array([[4, 3]])

    assert np.array_equal(output, expected)
    assert output.shape == expected.shape

def test_dense_backward():
    """
    Function to test the dense layer's backward pass.
    """
    dense = Dense(3, 2)

    dense.W = np.array([
        [2, 1],
        [1, 1],
        [1, 1]
    ])
    dense.b = np.array([[0, 0]])

    X = np.array([[1, 1, 1]])
    dense.forward(X)

    grad_output = np.array([[2, 3]])

    grad_input = dense.backward(grad_output)

    expected_dW = np.array([
        [2, 3],
        [2, 3],
        [2, 3]
    ])

    expected_db = np.array([[2, 3]])

    expected_dX = np.array([[7, 5, 5]])

    assert np.array_equal(dense.dW, expected_dW)
    assert np.array_equal(dense.db, expected_db)
    assert np.array_equal(grad_input, expected_dX)

def test_dense_forward_batch():
    """
    Function to test the dense layer's forward pass with a batched input.
    """
    dense = Dense(3, 2)

    dense.W = np.array([
        [2, 1],
        [1, 1],
        [1, 1]
    ])
    dense.b = np.array([[0, 0]])

    X = np.array([
        [1, 1, 1],
        [2, 2, 2]
    ])#A batch of two inputs

    output = dense.forward(X)

    expected = np.array([
        [4, 3],
        [8, 6]
    ])

    assert np.array_equal(output, expected)
    assert output.shape == expected.shape


def test_dense_backward_batch():
    """
    Function to test the dense layer's backward pass with a batched input.
    """
    dense = Dense(3, 2)

    dense.W = np.array([
        [2, 1],
        [1, 1],
        [1, 1]
    ])
    dense.b = np.array([[0, 0]])

    X = np.array([
        [1, 1, 1],
        [2, 2, 2]
    ])

    dense.forward(X)

    grad_output = np.array([
        [2, 3],
        [1, 1]
    ])

    grad_input = dense.backward(grad_output)

    expected_dW = np.array([
        [4, 5],
        [4, 5],
        [4, 5]
    ])

    expected_db = np.array([[3, 4]])

    expected_dX = np.array([
        [7, 5, 5],
        [3, 2, 2]
    ])

    assert np.array_equal(dense.dW, expected_dW)
    assert np.array_equal(dense.db, expected_db)
    assert np.array_equal(grad_input, expected_dX)