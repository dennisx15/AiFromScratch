from .base import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Updates parameters using:
        param = param - lr * grad
    """

    def __init__(self, lr):
        """
        :param lr: float
            Learning rate (step size for updates)
        """
        self.lr = lr

    def step(self, params):
        """
        Apply SGD update to each parameter.

        :param params: list of (param, grad)
        """
        # TODO:
        # loop through params
        # update each param using its gradient
        for param, grad in params:
            if grad is None:
                continue
            param -= self.lr * grad