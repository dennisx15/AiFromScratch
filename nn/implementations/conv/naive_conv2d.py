from nn.implementations.conv.base import ConvImplementation
from nn.backend import xp

class Naive(ConvImplementation):

    def get_patch_bounds(self, i, j, layer):
        """
        gets the bounds of a patch in a certain step of the convolution
        :param i: row index
        :param j: column index
        :return: the mapping of the inputs the weight interacted with
        """
        h_start = i * layer.stride
        h_end = i  * layer.stride + layer.kernel_size
        w_start = j * layer.stride
        w_end = j * layer.stride + layer.kernel_size
        return (h_start, h_end, w_start, w_end)


    def extract_patch(self, batch, i, j, X, layer):
        """

        :param patch_bounds: the mapping of the inputs the weight interacted with
        :param X: The input to apply the extraction to
        :return: The extracted patch
        """

        h_start, h_end, \
            w_start, w_end = \
            self.get_patch_bounds(i, j, layer)

        return X[
            batch,
            :,
            h_start:h_end,
            w_start:w_end
        ]

    def forward(self, layer, X):
        """
        Forward pass
        :param X: The input, some kind of tensor
        :return: The output matrix after forward pass
        """
        assert X.ndim == 4, "Conv2D expects input shape (B, C, H, W)"
        assert X.shape[1] == layer.in_channels, "Input channels do not match"

        B, OC, out_h, out_w = layer.calculate_output_shape(X)

        #aplly padding
        if layer.padding > 0:
            X = xp.pad(
                X,
                (
                    (0, 0),  # batch
                    (0, 0),  # channels
                    (layer.padding, layer.padding),
                    (layer.padding, layer.padding)
                )
            )
        layer.input = X
        output_matrix = xp.zeros((B, OC, out_h, out_w))

        for batch in range(B):
            for out_c in range(OC):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = self.extract_patch(
                            batch,
                            i,
                            j,
                            X,
                            layer
                        )

                        output_matrix[batch, out_c, i, j] = (
                                xp.sum(
                                    patch * layer.W[out_c]
                                )
                                + layer.b[out_c]
                        )
        return output_matrix

    def backward(self, layer, grad_output):
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

        layer.dW.fill(0)
        layer.db.fill(0)

        dX = xp.zeros_like(layer.input)

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
                            self.get_patch_bounds(i, j, layer)

                        # Input patch used during
                        # forward pass

                        patch = layer.input[
                            batch,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        # Weight gradients
                        # dL/dW += dL/dP * dP/dW
                        # dP/dW = patch

                        layer.dW[out_c] += (grad * patch)

                        layer.db[out_c] += grad # Bias gradients

                        # Input gradients
                        # Distribute gradient back
                        # through the kernel

                        dX[
                            batch,
                            :,
                            h_start:h_end,
                            w_start:w_end
                        ] += (grad * layer.W[out_c])

        return dX

