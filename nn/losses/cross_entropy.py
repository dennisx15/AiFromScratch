from .base import Loss
from nn.backend import xp

class CrossEntropyLoss(Loss):

    def forward(self, logits, y_true):
        """
        :param logits: raw outputs (no softmax yet)
        :param y_true: integer class labels
        """

        # softmax inside loss
        logits_shifted = logits - xp.max(logits, axis=1, keepdims=True)
        exp = xp.exp(logits_shifted)
        self.probs = exp / xp.sum(exp, axis=1, keepdims=True)

        self.y_true = y_true

        batch_size = logits.shape[0]

        # pick correct class probabilities
        correct_probs = self.probs[xp.arange(batch_size), y_true]

        loss = -xp.log(correct_probs).mean()

        return loss

    def backward(self):
        batch_size = self.probs.shape[0]

        grad = self.probs.copy()
        grad[xp.arange(batch_size), self.y_true] -= 1
        grad /= batch_size

        return grad