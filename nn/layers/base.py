class Layer:
    """
    Abstract base class for all neural network layers.
    """

    def __init__(self):
        self.input = None
        self.training = True  # for layers like dropout later

    def forward(self, X):
        """
        Forward pass: compute output from input
        :param X: input to the layer
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Backward pass: compute gradient w.r.t input
        :param grad_output: the gradient of the loss with respect to this layer’s output
        """
        raise NotImplementedError

    def parameters(self):
        """
        Return list of (param, grad) tuples
        """
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False