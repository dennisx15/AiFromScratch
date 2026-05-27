from nn.implementations.conv.base import ConvImplementation
from nn.implementations.conv.naive_conv2d import Naive
from nn.backend import xp

class Im2Col(ConvImplementation):
    """
    Im2Col implementation of convolutional network.
    Turns images into columns and does forward and backward pass as a matrix operation
    """

    def __init__(self):
        self.cache_h = None #Stores the computed column indeces to reuse over and over
        self.cache_w = None #Stores the computed row indeces to reuse over and over

        self.input = None
        self.out_h = None
        self.out_w = None
        self.B = None

    def cache_indeces(self, layer, out_h, out_w):
        """
        Computes the cache shape to be used in forward passes to improve speed
        :param layer: the convolutional neural network itself is passed into here
        :param X: input
        :return: the new shape of the cache
        """
        all_h = []
        all_w = []

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * layer.stride
                w_start = j * layer.stride

                h_indices = xp.arange(
                    h_start,
                    h_start + layer.kernel_size
                )

                w_indices = xp.arange(
                    w_start,
                    w_start + layer.kernel_size
                )

                hh, ww = xp.meshgrid(
                    h_indices,
                    w_indices,
                    indexing="ij"
                )
                all_h.append(hh)
                all_w.append(ww)
        self.cache_h = xp.array(all_h)
        self.cache_w = xp.array(all_w)

    def im2col(self, layer, X):

        assert X.ndim == 4, "Conv2D expects input shape (B, C, H, W)"
        assert X.shape[1] == layer.in_channels, "Input channels do not match"
        B, _, out_h, out_w = layer.calculate_output_shape(X)
        self.B = B
        self.out_h = out_h
        self.out_w = out_w

        # apply padding
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
        self.input = X


        if self.cache_h is None or self.cache_w is None:
            self.cache_indeces(layer, out_h, out_w)

        flat_kernel = layer.W.reshape(
            layer.out_channels,
            -1
        )

        outputs = []
        for batch in range(B):
            patch = X[
                batch,
                :,
                self.cache_h,
                self.cache_w
            ]

            # Flatten patch
            flat_patch = patch.reshape(
                patch.shape[0],
                -1
            )

            # Matrix multiply against ALL kernels
            output = flat_patch @ flat_kernel.T + layer.b
            #print(output)
            #print(output.shape)
            outputs.append(output)


        outputs = xp.array(outputs)
        outputs = outputs.reshape(
            B,
            out_h,
            out_w,
            layer.out_channels
        )

        outputs = outputs.transpose(0, 3, 1, 2)

        return outputs

    def forward(self, layer, X):
        """
        :param layer: the convolutional neural network itself is passed into here
        :param X: input
        :return: the result of forward pass
        """
        return self.im2col(layer, X)

    def backward(self, layer, grad_output):
        """
        :param layer: the convolutional neural network itself is passed into here
        :param grad_output: Gradient of the loss with respect to this layer's output.
        :return: Gradient of the loss with respect to the input.
        """

        return self.col2im(layer, grad_output)

    def compute_dW(self, batch, grad_output, layer):
        patch = self.input[
            batch,
            :,
            self.cache_h,
            self.cache_w
        ] # Create patches

        # Flatten patch
        flat_patch = patch.reshape(
            patch.shape[0],
            -1
        )

        flat_grad_out = grad_output[
            batch
        ].reshape(
            layer.out_channels,
            -1
        )

        # Matrix multiply against ALL kernels
        output = flat_patch.T @ flat_grad_out.T  # the gradient of the loss with respect to weights

        output = output.T.reshape(
            layer.out_channels,
            layer.in_channels,
            layer.kernel_size,
            layer.kernel_size
        )

        return output

    def compute_dW_across_entire_batch(self, grad_output, layer):
        patch = self.input[
            :,
            :,
            self.cache_h,
            self.cache_w
        ] # Create patches

        # Flatten patch
        flat_patch = patch.reshape(
            self.B,
            self.out_h * self.out_w,
            -1
        )

        flat_grad_out = grad_output.reshape(
            self.B,
            layer.out_channels,
            -1
        ).transpose(0, 2, 1)

        output = flat_patch.transpose(0, 2, 1) @ flat_grad_out

        output = xp.sum(output, axis=0)
        output = output.T.reshape(
            layer.out_channels,
            layer.in_channels,
            layer.kernel_size,
            layer.kernel_size
        )

        return output




    def compute_dX(self, layer, grad_output, batch, dX):
        W_flat = layer.W.reshape(
            layer.out_channels,
            -1
        )

        flat_grad_out = grad_output[
            batch
        ].reshape(
            layer.out_channels,
            -1
        )

        dX_cols = flat_grad_out.T @ W_flat

        dX_cols = dX_cols.reshape(
            self.out_h * self.out_w,
            layer.in_channels,
            layer.kernel_size,
            layer.kernel_size
        )

        dX_cols = dX_cols.transpose(1, 0, 2, 3)

        xp.add.at(
            dX[batch],
            (
                slice(None),
                self.cache_h,
                self.cache_w
            ),
            dX_cols
        )

    def compute_dX_across_entire_batch(self, grad_output, layer):
        dX = xp.zeros_like(layer.input)

        W_flat = layer.W.reshape(
            layer.out_channels,
            -1
        )
        flat_grad_out = grad_output.reshape(
            self.B,
            layer.out_channels,
            -1
        ).transpose(0, 2, 1)

        dX_cols = flat_grad_out @ W_flat

        dX_cols = dX_cols.reshape(
            self.B,
            self.out_h * self.out_w,
            layer.in_channels,
            layer.kernel_size,
            layer.kernel_size
        )

        dX_cols = dX_cols.transpose(0, 2, 1, 3, 4)

        for batch in range(self.B):
            xp.add.at(
                dX[batch],
                (
                    slice(None),
                    self.cache_h,
                    self.cache_w
                ),
                dX_cols[batch]
            )

        if layer.padding > 0:
            dX = dX[
                :,
                :,
                layer.padding:-layer.padding,
                layer.padding:-layer.padding
            ]


        return dX



    def col2im(self, layer, grad_output):
        """
        performs backward pass using col2im
        :param layer: the convolutional neural network itself is passed into here
        :param grad_output: the gradient of the loss with respect to this layer's output.
        :return: gradient of the loss with respect to the input.
        """

        #assert self.cache_h and self.cache_w, "Can't do backward pass without performing a forward pass first"

        #dX = xp.zeros_like(layer.input)

        dW = self.compute_dW_across_entire_batch(grad_output, layer)

        layer.dW += dW
        # layer.db += xp.sum(
        #     grad_output[batch],
        #     axis=(1, 2)
        # )

        layer.db += xp.sum(
             grad_output,
             axis=(0, 2, 3)
         ) #vectorize across batch

        # for batch in range(self.B):
        #
        #
        #
        #
        #     self.compute_dX(layer, grad_output, batch, dX)

        # if layer.padding > 0:
        #     dX = dX[
        #         :,
        #         :,
        #         layer.padding:-layer.padding,
        #         layer.padding:-layer.padding
        #     ]
        #
        #
        # return dX

        return self.compute_dX_across_entire_batch(grad_output, layer)



# ============================================================
# Legacy / Reference Implementations
# ============================================================



    def _legacy_im2col(self, layer, X):
        """
        legacy implementation of im2col kept for debugging, benchmarking and historical reference
         :param layer: the convolutional neural network itself is passed into here
         :param X: input
         :return: result of forward pass
         """

        assert X.ndim == 4, "Conv2D expects input shape (B, C, H, W)"
        assert X.shape[1] == layer.in_channels, "Input channels do not match"

        B, OC, out_h, out_w = layer.calculate_output_shape(X)

        # apply padding
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
        patches = []

        # Flatten kernels
        flat_kernel = layer.W.reshape(
            layer.out_channels,
            -1
        )

        for batch in range(B):

            for i in range(out_h):

                for j in range(out_w):
                    h_start = i * layer.stride
                    w_start = j * layer.stride

                    # Extract patch
                    patch = X[
                        batch,
                        :,
                        h_start:h_start + layer.kernel_size,
                        w_start:w_start + layer.kernel_size
                    ]

                    # Flatten patch
                    flat_patch = patch.flatten()

                    # Matrix multiply against ALL kernels
                    output = flat_patch @ flat_kernel.T

                    patches.append(output)

        # Shape:
        # (num_patches, out_channels)
        patches = xp.array(patches)

        # Reshape into CNN output layout
        patches = patches.reshape(
            B,
            out_h,
            out_w,
            layer.out_channels
        )

        # Convert:
        # (B, H, W, OC)
        #
        # to:
        # (B, OC, H, W)
        patches = patches.transpose(
            0,
            3,
            1,
            2
        )

        return patches
