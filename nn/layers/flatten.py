from .base import Layer
from nn.backend import xp


class Flatten(Layer):
    """
    Flatten layer. Used at the end of a convolutional layer to pass it into a dense layer.
    """
    def forward(self, X):
        """
        Flatten input except batch dimension.

        Args:
            X (xp.ndarray):
                Shape (B, ...)

        Returns:
            xp.ndarray:
                Shape (B, flattened_features)
        """

        self.input_shape = X.shape

        return X.reshape(X.shape[0], -1)

    def backward(self, grad_output):
        """
        Reshape gradient back to original input shape.
        """

        return grad_output.reshape(self.input_shape)