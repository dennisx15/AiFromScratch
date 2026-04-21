from ..model import Model
def MLP(input_dim, hidden_dims, output_dim):
    """
    Build a multi-layer perceptron.

    :param input_dim: int
        Number of input features.

    :param hidden_dims: list[int]
        Sizes of hidden layers.

    :param output_dim: int
        Number of output units.

    :return: Model
        A neural network composed of Dense + activation layers.
    """
    layers = []

    # TODO:
    # - loop through hidden_dims
    # - add Dense + Activation
    # - finish with output layer

    return Model(layers)