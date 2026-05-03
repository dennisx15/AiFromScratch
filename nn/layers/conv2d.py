from .base import Layer
from nn.backend import xp


class Conv2D(Layer):
    """
    Conv2D layer. Does forward and backward pass
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: kernel size
        """
        super().__init__()

    def forward(self, X):
        ...

    def backward(self, grad_output):
        ...