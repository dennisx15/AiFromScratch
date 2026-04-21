class Optimizer:
    """
    Abstract base class for all optimizers.

    An optimizer updates model parameters using their gradients.
    """

    def step(self, params):
        """
        Update parameters in-place.

        :param params: list[tuple]
            List of (param, grad) tuples.

            - param: np.ndarray
                The parameter to update (e.g., weights, bias)

            - grad: np.ndarray
                Gradient of loss w.r.t that parameter (same shape as param)

        :return: None
        """
        raise NotImplementedError