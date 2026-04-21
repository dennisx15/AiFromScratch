from .base import Loss

class MSE(Loss):
    """
    Mean Squared Error loss.

    Measures average squared difference between predictions and targets.
    Commonly used for regression tasks.
    """

    def forward(self, y_pred, y_true):
        """
        Store inputs and compute MSE loss.

        :param y_pred: np.ndarray
            Predicted values from the model.

        :param y_true: np.ndarray
            Actual target values.

        :return: float
            Mean squared error over the batch.
        """
        self.y_pred = y_pred
        self.y_true = y_true

        # TODO: implement MSE formula
        ...

    def backward(self):
        """
        Compute gradient of MSE loss with respect to predictions.

        :return: np.ndarray
            Gradient dL/dy_pred.
            Same shape as y_pred.
        """
        # TODO: derive and implement gradient
        ...