import numpy as np
from nn.backend import xp
class Model:
    """
    Represents a neural network as a sequence of layers.
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def save(self, path):

        params = self.parameters()

        data = {}
        for i, (param, _) in enumerate(params):
            # convert to numpy if on GPU
            try:
                import cupy as cp
                if isinstance(param, cp.ndarray):
                    param = cp.asnumpy(param)
            except:
                pass

            data[f"param_{i}"] = param

        np.savez(path, **data)

    def save_loss_and_acc(self, path, loss, acc):
        np.savez(path, loss=loss, acc=acc)

    def load(self, path):

        data = np.load(path)

        params = self.parameters()

        for i, (param, _) in enumerate(params):
            param[:] = xp.array(data[f"param_{i}"])

    def load_loss_and_acc(self, path):
        """
        :param path: path to load loss & acc
        :return: (loss, acc)
        """
        data = np.load(path)
        loss = data["loss"]
        acc = data["acc"]
        return loss, acc