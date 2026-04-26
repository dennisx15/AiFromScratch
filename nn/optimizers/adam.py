from .base import Optimizer
from nn.backend import xp

class Adam(Optimizer):


    """
    Adam optimizer
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """

        :param params: list of (param, grad)
        :param lr: learning rate
        :param beta1: controls the momentum for mean
        :param beta2: controls the momentum for variance
        :param eps:
        """
        self.mean = None
        self.variance = None
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.lr = lr

    def step(self, params):
        if self.mean is None:
            self.mean = [xp.zeros_like(p) for p, _ in params]
            self.variance = [xp.zeros_like(p) for p, _ in params]

        self.t += 1

        for i, (param, grad) in enumerate(params):
            # running averages
            self.mean[i] = self.beta1 * self.mean[i] + (1 - self.beta1) * grad
            self.variance[i] = self.beta2 * self.variance[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.mean[i] / (1 - self.beta1 ** self.t)
            v_hat = self.variance[i] / (1 - self.beta2 ** self.t)
            param[:] -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)