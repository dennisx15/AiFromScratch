from .base import Layer
from nn.backend import xp


class Conv2D(Layer):
    """
    Conv2D layer. Does forward and backward pass
    """
    def __init__(self, out_channels, in_channels=1, kernel_size=3, stride=1, padding=0):
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

    def calculate_output_shape(self, X):
        """
        Calculate output shape
        :param X: the input, some kind of tensor
        :return: the dimensions of the output
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
        assert X.ndim == 4, "Conv2D expects input shape (B, C, H, W)"
        B, OC, out_h, out_w = self.calculate_output_shape(X)
        output_matrix = xp.zeros((B, OC, out_h, out_w))

        for batch in range(B):
            for out_c in range(OC):
                for i in range(out_h):
                    for j in range(out_w):
                        output_matrix[batch, out_c, i, j] = \
                            xp.sum(X[batch, :, i:i+self.kernel_size, j:j+self.kernel_size] \
                        * self.W[out_c]) + self.b[out_c]
        return output_matrix

    def backward(self, grad_output):
        ...

