from nn.backend import xp
from nn.implementations.conv.im2col import Im2Col
from nn.layers.conv2d import Conv2D

def test_im2col():
    conv2d = Conv2D(kernel_size=2, out_channels=2, in_channels=1)
    im2col = Im2Col()

    conv2d.W[0] = xp.array([
        [1, 1],
        [1, 1]
    ])

    # Second kernel
    conv2d.W[1] = xp.array([
        [1, 2],
        [3, 4]
    ])

    X = xp.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ])

    print(im2col.im2col(conv2d, X))
    print(im2col.im2col(conv2d, X).shape)


def test_caching():
    conv2d = Conv2D(kernel_size=2, out_channels=2, in_channels=1)
    im2col = Im2Col()

    conv2d.W[0] = xp.array([
        [1, 1],
        [1, 1]
    ])

    # Second kernel
    conv2d.W[1] = xp.array([
        [1, 2],
        [3, 4]
    ])

    X = xp.array([
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ])

    im2col.cache_indeces(conv2d, X)
    #print(im2col.cache_w)
    #print(im2col.cache_h)
    #print(X[0,:,im2col.cache_h,im2col.cache_w])
    print(im2col.new_im2col(conv2d, X))
    print(im2col.im2col(conv2d, X))

