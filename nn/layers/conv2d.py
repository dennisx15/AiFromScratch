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


    def get_patch_bounds(self, i, j):
        """
        :param i: row index
        :param j: column index
        :return: the mapping of the inputs the weight interacted with
        """
        h_start = i * self.stride
        h_end = i  * self.stride + self.kernel_size
        w_start = j * self.stride
        w_end = j * self.stride + self.kernel_size
        return (h_start, h_end, w_start, w_end)


    def extract_patch(self, batch, i, j, X):
        """

        :param patch_bounds: the mapping of the inputs the weight interacted with
        :param X: The input to apply the extraction to
        :return: The extracted patch
        """

        h_start, h_end, \
            w_start, w_end = \
            self.get_patch_bounds(i, j)

        return X[
            batch,
            :,
            h_start:h_end,
            w_start:w_end
        ]

    def forward(self, X):
        """
        Forward pass
        :param X: The input, some kind of tensor
        :return: The output matrix after forward pass
        """
        assert X.ndim == 4, "Conv2D expects input shape (B, C, H, W)"
        assert X.shape[1] == self.in_channels, "Input channels do not match"

        #aplly padding
        if self.padding > 0:
            X = xp.pad(
                X,
                (
                    (0, 0),  # batch
                    (0, 0),  # channels
                    (self.padding, self.padding),
                    (self.padding, self.padding)
                )
            )

        self.input = X

        self.input = X
        B, OC, out_h, out_w = self.calculate_output_shape(X)
        output_matrix = xp.zeros((B, OC, out_h, out_w))

        for batch in range(B):
            for out_c in range(OC):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = self.extract_patch(
                            batch,
                            i,
                            j,
                            X
                        )

                        output_matrix[batch, out_c, i, j] = (
                                xp.sum(
                                    patch * self.W[out_c]
                                )
                                + self.b[out_c]
                        )
        return output_matrix

    def backward(self, grad_output):
        """
        Backward pass for Conv2D.

        :param grad_output:
            Gradient of the loss with respect to
            this layer's output.

            Shape:
            (B, OC, out_h, out_w)

        :return:
            Gradient of the loss with respect
            to the input.

            Shape:
            (B, IC, H, W)
        """

        # -------------------------
        # Initialize gradients
        # -------------------------

        self.dW.fill(0)
        self.db.fill(0)

        dX = xp.zeros_like(self.input)

        # -------------------------
        # Shapes
        # -------------------------

        B, OC, out_h, out_w = grad_output.shape

        # -------------------------
        # Main backward loops
        # -------------------------

        for batch in range(B):

            for out_c in range(OC):

                for i in range(out_h):

                    for j in range(out_w):
                        # Current upstream gradient
                        # dL/dP

                        grad = grad_output[
                            batch,
                            out_c,
                            i,
                            j
                        ]

                        # Spatial bounds

                        h_start, h_end, \
                            w_start, w_end = \
                            self.get_patch_bounds(i, j)

                        # Input patch used during
                        # forward pass

                        patch = self.input[
                            batch,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        # Weight gradients
                        # dL/dW += dL/dP * dP/dW
                        # dP/dW = patch

                        self.dW[out_c] += (grad * patch)

                        self.db[out_c] += grad # Bias gradients

                        # Input gradients
                        # Distribute gradient back
                        # through the kernel

                        dX[
                            batch,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ] += (grad * self.W[out_c])

        return dX

