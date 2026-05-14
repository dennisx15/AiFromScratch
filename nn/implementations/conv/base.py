from abc import ABC, abstractmethod


class ConvImplementation(ABC):
    """
    Abstract base class for all convolution
    implementations.

    Implementations define HOW convolution
    executes, while Conv2D defines WHAT
    a convolution layer is.
    """

    @abstractmethod
    def forward(self, layer, X):
        """
        Forward pass.

        :param layer:
            The Conv2D layer using this implementation.

        :param X:
            Input tensor.

        :return:
            Output tensor.
        """
        pass

    @abstractmethod
    def backward(self, layer, grad_output):
        """
        Backward pass.

        :param layer:
            The Conv2D layer using this implementation.

        :param grad_output:
            Gradient of the loss with respect
            to this layer's output.

        :return:
            Gradient with respect to input.
        """
        pass