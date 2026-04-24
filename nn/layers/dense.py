from .base import Layer
from nn.backend import xp
class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        """
        The Dense layer, a building block of neural networks.
        :param input_dim: input dimension -> The dimension of the input vector.
        :param output_dim: output dimension -> The dimension of the output vector.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize weights
        self.W = xp.random.randn(input_dim, output_dim) * 0.01
        self.b = xp.zeros((1, output_dim))
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)
    def forward(self, X):
        """
        Forward pass: compute output from input
        :param X: input to the layer
        """
        self.input = X
        self.output = X @ self.W + self.b
        return self.output

    def backward(self, grad_output):
        """
        Backward pass: compute gradient w.r.t input
        :param grad_output: the gradient of the loss with respect to this layer’s output
        """
        self.dW = self.input.T @ grad_output#-> Gradient of loss w.r.t weights (dL/dW)
        self.db = grad_output.sum(axis=0, keepdims=True)
        return grad_output @ self.W.T

    def parameters(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]
