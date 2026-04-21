class Loss:
    """
    Abstract base class for all loss functions.

    A loss function measures how far the model's predictions are from the true targets.
    It is also responsible for producing the initial gradient used in backpropagation.
    """

    def forward(self, y_pred, y_true):
        """
        Compute the loss value.

        :param y_pred: np.ndarray
            Model predictions (output of the final layer).
            Shape: (batch_size, output_dim)

        :param y_true: np.ndarray
            Ground truth labels corresponding to y_pred.
            Shape: (batch_size, output_dim)

        :return: float
            Scalar loss value representing prediction error.
        """
        raise NotImplementedError

    def backward(self):
        """
        Compute gradient of the loss with respect to predictions.

        :return: np.ndarray
            Gradient dL/dy_pred (same shape as y_pred).
            This is the starting point of backpropagation.
        """
        raise NotImplementedError