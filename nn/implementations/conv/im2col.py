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
        naive = Naive()
        return naive.backward(layer, grad_output)

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
