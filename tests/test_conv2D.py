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

def test_conv2d_forward_stride():

    """
    Tests the forward pass of the conv2D layer
    """
    conv2d = Conv2D(kernel_size=2, out_channels=2, stride=2)

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
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ]
        ]
    ])

    expected = xp.array([
        [
            [
                [14, 22],
                [46, 54]
            ],
            [
                [44, 64],
                [124, 144]
            ]
        ]
    ])

    output = conv2d.forward(X)

    assert xp.array_equal(output, expected)

def test_conv2d_backward():
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
    ], dtype=xp.float64)
    grad_output = xp.array([
        [
            [
                [1, 1],
                [1, 1]
            ],
            [
                [1, 1],
                [1, 1]
            ]
        ]
    ], dtype=xp.float64)

    expected = xp.array([
        [
            [
                [12, 16],
                [24, 28]
            ]
        ],

        [
            [
                [12, 16],
                [24, 28]
            ]
        ]
    ], dtype=xp.float64)

    conv2d.forward(X)
    conv2d.backward(grad_output)
    assert xp.array_equal(xp.array(conv2d.dW), expected)

def test_conv2d_backward_stride():
    conv2d = Conv2D(kernel_size=2, out_channels=2, stride=2)

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
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ]
        ]
    ], dtype=xp.float64)
    grad_output = xp.array([
        [
            [
                [1, 1],
                [1, 1]
            ],
            [
                [1, 1],
                [1, 1]
            ]
        ]
    ], dtype=xp.float64)

    expected = xp.array([
        [
            [
                [24, 28],
                [40, 44]
            ]
        ],

        [
            [
                [24, 28],
                [40, 44]
            ]
        ]
    ], dtype=xp.float64)

    conv2d.forward(X)
    conv2d.backward(grad_output)
    assert xp.array_equal(xp.array(conv2d.dW), expected)