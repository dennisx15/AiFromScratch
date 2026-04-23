from .base import Layer
import numpy as np
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        :param X: np.ndarray
    Input tensor of shape (batch_size, features).
        :return: The vector after applying the sigmoid activation function.
        """
        self.output = 1/(1+np.exp(-X))
        return self.output

    def backward(self, grad_output):
        """
        :param grad_output: np.ndarray
    Gradient of the loss with respect to this layer's output (dL/dY).
        :return: The gradient of the output vector after applying the sigmoid activation function.
        """
        return grad_output * self.output * (1 - self.output)

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        :param X: np.ndarray
    Input tensor of shape (batch_size, features).
        :return: The vector after applying the relu activation function.
        """
        self.input = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        """
        :param grad_output: np.ndarray
    Gradient of the loss with respect to this layer's output (dL/dY).
        :return: The gradient of the output vector after applying the relu activation function.
        """

        return grad_output * (self.input > 0).astype(float)

class Swish(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        :param X: np.ndarray
    Input tensor of shape (batch_size, features).
        :return: The vector after applying the swish activation function.
        """
        self.input = X
        self.sigmoid = 1 / (1 + np.exp(-X))
        return X * self.sigmoid


    def backward(self, grad_output):
        """
        :param grad_output: np.ndarray
    Gradient of the loss with respect to this layer's output (dL/dY).
        :return: The gradient of the output vector after applying the swish activation function.
        """
        return grad_output * (self.sigmoid + self.input * self.sigmoid * (1 - self.sigmoid))