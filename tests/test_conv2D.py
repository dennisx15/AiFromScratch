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

def test_conv2d_forward_padded():

    """
    Tests the forward pass of the conv2D layer
    """
    conv2d = Conv2D(kernel_size=2, out_channels=2, padding=1)

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
                [1, 3, 5, 3],
                [5, 12, 16, 9],
                [11, 24, 28, 15],
                [7, 15, 17, 9]
            ],

            [
                [4, 11, 18, 9],
                [18, 37, 47, 21],
                [36, 67, 77, 33],
                [14, 23, 26, 9]
            ]
        ]
    ])

    output = conv2d.forward(X)
    print(output)

    assert xp.array_equal(output, expected)


def test_two_layer_conv_forward():
    """
    Tests the forward pass of a
    2-layer convolution pipeline.
    """

    # ---------------------------------
    # Layer 1
    # ---------------------------------

    conv1 = Conv2D(
        kernel_size=2,
        out_channels=2
    )

    conv1.W[0] = xp.array([
        [
            [1, 1],
            [1, 1]
        ]
    ], dtype=xp.float64)

    conv1.W[1] = xp.array([
        [
            [1, 2],
            [3, 4]
        ]
    ], dtype=xp.float64)

    # ---------------------------------
    # Layer 2
    # ---------------------------------

    conv2 = Conv2D(
        kernel_size=2,
        in_channels=2,
        out_channels=1
    )

    conv2.W[0] = xp.array([

        # channel 0
        [
            [1, 1],
            [1, 1]
        ],

        # channel 1
        [
            [1, 1],
            [1, 1]
        ]

    ], dtype=xp.float64)

    # ---------------------------------
    # Input
    # ---------------------------------

    X = xp.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ], dtype=xp.float64)

    # ---------------------------------
    # Forward pass
    # ---------------------------------

    out1 = conv1.forward(X)

    expected_out1 = xp.array([
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
    ], dtype=xp.float64)

    assert xp.array_equal(out1, expected_out1)

    out2 = conv2.forward(out1)

    expected_out2 = xp.array([
        [
            [
                [308]
            ]
        ]
    ], dtype=xp.float64)

    assert xp.array_equal(out2, expected_out2)

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

def test_conv2d_backward_padded():
    """
     Tests the backward pass of the conv2D layer with padding
     """
    conv2d = Conv2D(kernel_size=2, out_channels=2, padding=1)

    # First kernel
    conv2d.W[0] = xp.array([
        [1, 1],
        [1, 1]
    ], dtype=xp.float64)

    # Second kernel
    conv2d.W[1] = xp.array([
        [1, 2],
        [3, 4]
    ], dtype=xp.float64)

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

    grad_output = xp.ones((1,2,4,4))

    expected = xp.array([
        [
            [
                [45, 45],
                [45, 45]
            ]
        ],

        [
            [
                [45, 45],
                [45, 45]
            ]
        ]
    ], dtype=xp.float64)

    conv2d.forward(X)
    conv2d.backward(grad_output)


    assert xp.array_equal(xp.array(conv2d.dW), expected)


def test_two_layer_conv_backward():
    """
    Tests the backward pass of a
    2-layer convolution pipeline.
    """

    # ---------------------------------
    # Layer 1
    # ---------------------------------

    conv1 = Conv2D(
        kernel_size=2,
        out_channels=2
    )

    conv1.W[0] = xp.array([
        [
            [1, 1],
            [1, 1]
        ]
    ], dtype=xp.float64)

    conv1.W[1] = xp.array([
        [
            [1, 2],
            [3, 4]
        ]
    ], dtype=xp.float64)

    # ---------------------------------
    # Layer 2
    # ---------------------------------

    conv2 = Conv2D(
        kernel_size=2,
        in_channels=2,
        out_channels=1
    )

    conv2.W[0] = xp.array([

        # channel 0
        [
            [1, 1],
            [1, 1]
        ],

        # channel 1
        [
            [1, 1],
            [1, 1]
        ]

    ], dtype=xp.float64)

    # ---------------------------------
    # Input
    # ---------------------------------

    X = xp.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ], dtype=xp.float64)

    # ---------------------------------
    # Forward pass
    # ---------------------------------

    out1 = conv1.forward(X)
    out2 = conv2.forward(out1)

    # ---------------------------------
    # Backward pass
    # ---------------------------------

    grad_output = xp.array([
        [
            [
                [1]
            ]
        ]
    ], dtype=xp.float64)

    grad1 = conv2.backward(grad_output)

    expected_dW_conv2 = xp.array([
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
    ], dtype=xp.float64)

    assert xp.array_equal(
        conv2.dW,
        expected_dW_conv2
    )

    conv1.backward(grad1)

    expected_dW_conv1 = xp.array([

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

    assert xp.array_equal(
        conv1.dW,
        expected_dW_conv1
    )

def test_conv2d_dx():
    """
    Tests the input gradients (dX)
    of the Conv2D backward pass.
    """

    conv2d = Conv2D(
        kernel_size=2,
        out_channels=1
    )

    # ---------------------------------
    # Kernel
    # ---------------------------------

    conv2d.W[0] = xp.array([
        [
            [1, 2],
            [3, 4]
        ]
    ], dtype=xp.float64)

    # ---------------------------------
    # Input
    # ---------------------------------

    X = xp.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ], dtype=xp.float64)

    # ---------------------------------
    # Forward
    # ---------------------------------

    conv2d.forward(X)

    # ---------------------------------
    # Upstream gradients
    # ---------------------------------

    grad_output = xp.array([
        [
            [
                [1, 1],
                [1, 1]
            ]
        ]
    ], dtype=xp.float64)

    # ---------------------------------
    # Backward
    # ---------------------------------

    dX = conv2d.backward(grad_output)

    # ---------------------------------
    # Expected dX
    # ---------------------------------

    expected = xp.array([
        [
            [
                [1, 3, 2],
                [4, 10, 6],
                [3, 7, 4]
            ]
        ]
    ], dtype=xp.float64)

    assert xp.array_equal(dX, expected)