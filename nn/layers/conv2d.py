from .base import Layer
from nn.backend import xp
from nn.implementations.conv.naive_conv2d import Naive


class Conv2D(Layer):
    """
    Conv2D layer. Does forward and backward pass
    """
    def __init__(self, out_channels, in_channels=1, kernel_size=3, stride=1, padding=0, implementation=Naive()):
        """
        :param in_channels: The number of channels the input has. 1 for grayscale
        :param out_channels: The number of channels of the output, or the number of kernels
        :param kernel_size: kernel size
        """
        super().__init__()

        fan_in = in_channels * kernel_size * kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # randn samples from N(0, 1).
        # Without scaling, activation and gradient magnitudes
        # can grow or shrink across layers.
        # He initialization helps stabilize this.
        self.W = xp.random.randn(out_channels, in_channels, kernel_size, kernel_size)* xp.sqrt(2 / fan_in)
        self.b = xp.zeros(out_channels)
        #To keep track of gradients for later
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)
        self.input = None
        self.implementation = implementation

    def calculate_output_shape(self, X):
        """
        Calculate output shape
        :param X: the input, some kind of tensor
        :return: the dimensions of the output
        Note:
        Intended for internal Conv2D use only.

        """
        batch, channel, height, width = X.shape

        out_h = (
                (height + 2 * self.padding - self.kernel_size)
                // self.stride
                ) + 1

        out_w = (
                (width + 2 * self.padding - self.kernel_size)
                // self.stride
                ) + 1

        return (
            batch,
            self.out_channels,
            out_h,
            out_w
        )


    def forward(self, X):
        """
        Forward pass
        :param X: The input, some kind of tensor
        :return: The output matrix after forward pass
        """
        return self.implementation.forward(self, X)


    def backward(self, grad_output):
        """
        Backward pass for Conv2D.

        :param grad_output:
        Gradient of the loss with respect to
        this layer's output.
        :return:
            Gradient of the loss with respect
            to the input.
            """
        return self.implementation.backward(self, grad_output)

