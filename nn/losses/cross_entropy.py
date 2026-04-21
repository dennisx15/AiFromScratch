from .base import Loss

class CrossEntropy(Loss):
    """
    Cross-Entropy loss for classification.

    Typically used with probability outputs (e.g., after Softmax).
    Measures how well predicted probability distributions match true labels.
    """

    def forward(self, y_pred, y_true):
        """
        :param y_pred: np.ndarray
            Predicted probabilities for each class.
            Shape: (batch_size, num_classes)

        :param y_true: np.ndarray
            One-hot encoded true labels.
            Shape: (batch_size, num_classes)

        :return: float
            Average cross-entropy loss.
        """
        self.y_pred = y_pred
        self.y_true = y_true

        # TODO: implement cross-entropy
        ...

    def backward(self):
        """
        Compute gradient of cross-entropy loss.

        :return: np.ndarray
            Gradient dL/dy_pred.
        """
        # TODO: implement gradient
        ...