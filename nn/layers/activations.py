from .base import Layer
import numpy as np
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        :param X: The input vector
        :return: The vector after applying the sigmoid activation function.
        """
        ...
    def backward(self, grad_output):
        """
        :param grad_output: The gradient of the output vector.
        :return: The gradient of the output vector after applying the sigmoid activation function.
        """
        ...

class Relu(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        :param X: The input vector
        :return: The vector after applying the relu activation function.
        """
        ...

    def backward(self, grad_output):
        """
        :param grad_output: The gradient of the output vector.
        :return: The gradient of the output vector after applying the relu activation function.
        """
        ...

class Swish(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        :param X: The input vector
        :return: The vector after applying the swish activation function.
        """
        ...

    def backward(self, grad_output):
        """
        :param grad_output: The gradient of the output vector.
        :return: The gradient of the output vector after applying the swish activation function.
        """
        ...