from nn.backend import xp
from nn.layers.conv2d import Conv2D


def test_conv2d_forward():

    """
    Tests the forward pass of the conv2D layer
    """
    conv2d = Conv2D(kernel_size=2, out_channels=2)

    # First kernel
    conv2d.W[0] = xp.array([
        [1, 1],
        [1, 1]
    ])

    # Second kernel
    conv2d.W[1] = xp.array([
        [1, 2],
        [3, 4]
    ])

    # Input shape: (B, C, H, W)
    X = xp.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ])

    expected = xp.array([
        [
            [
                [12, 16],
                [24, 28]
            ],
            [
                [37, 47],
                [67, 77]
            ]
        ]
    ])

    output = conv2d.forward(X)

    assert xp.array_equal(output, expected)