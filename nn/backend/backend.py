import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

xp = np  # default


def set_device(device="cpu"):
    global xp

    if device == "cpu":
        xp = np

    elif device == "gpu":
        if cp is None:
            raise RuntimeError("CuPy is not installed")
        xp = cp

    else:
        raise ValueError("device must be 'cpu' or 'gpu'")