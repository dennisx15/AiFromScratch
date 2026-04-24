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
        import numpy as np

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

    def load(self, path):
        import numpy as np
        from nn.backend import xp

        data = np.load(path)

        params = self.parameters()

        for i, (param, _) in enumerate(params):
            param[:] = xp.array(data[f"param_{i}"])