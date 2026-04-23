import numpy as np
import pytest

from nn.layers.activations import Relu

def test_relu_forward():
    relu = Relu()
    assert relu.forward(np.ones((1, 1, 1))) == np.ones((1, 1, 1))
