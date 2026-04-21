class Model:
    """
    Represents a neural network as a sequence of layers.
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        """
        Pass input through all layers sequentially.
        """
        ...

    def backward(self, grad):
        """
        Backpropagate gradient through layers in reverse order.
        """
        ...

    def parameters(self):
        """
        Collect all parameters from all layers.
        """
        ...