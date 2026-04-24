from .base import Loss
import numpy as np

class CrossEntropyLoss(Loss):

    def forward(self, logits, y_true):
        """
        :param logits: raw outputs (no softmax yet)
        :param y_true: integer class labels
        """

        # softmax inside loss
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits_shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        self.y_true = y_true

        batch_size = logits.shape[0]

        # pick correct class probabilities
        correct_probs = self.probs[np.arange(batch_size), y_true]

        loss = -np.log(correct_probs).mean()

        return loss

    def backward(self):
        batch_size = self.probs.shape[0]

        grad = self.probs.copy()
        grad[np.arange(batch_size), self.y_true] -= 1
        grad /= batch_size

        return grad