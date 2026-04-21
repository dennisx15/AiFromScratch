from .base import Layer
import numpy as np
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
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))