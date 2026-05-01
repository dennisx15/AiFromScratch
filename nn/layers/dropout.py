from .base import Layer
from nn.backend import xp

class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, X):
        if not self.training:
            return X

        self.mask = (xp.random.rand(*X.shape) < self.p).astype(X.dtype)
        return (X * self.mask) / self.p

    def backward(self, grad_output):
        if not self.training:
            return grad_output

        return (grad_output * self.mask) / self.p #Zero out some gradients, and divide by p to have the same expected value