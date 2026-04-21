class Trainer:
    """
    Handles the training process of a model.

    Responsible for coordinating forward pass, loss computation,
    backward pass, and parameter updates.
    """

    def __init__(self, model, loss_fn, optimizer):
        """
        :param model: Model
            The neural network to train.

        :param loss_fn: Loss
            Loss function used to compute error.

        :param optimizer: Optimizer
            Optimizer used to update parameters.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, X, y):
        """
        Perform one training step (forward + backward + update).

        :param X: np.ndarray
            Input batch.

        :param y: np.ndarray
            Ground truth labels.

        :return: float
            Loss value for this batch.
        """

        # 1. Forward pass
        y_pred = self.model.forward(X)

        # 2. Compute loss
        loss = self.loss_fn.forward(y_pred, y)

        # 3. Get gradient from loss
        grad = self.loss_fn.backward()

        # 4. Backprop through model
        self.model.backward(grad)

        # 5. Update parameters
        self.optimizer.step(self.model.parameters())

        return loss

    def fit(self, X, y, epochs):
        """
        Train the model for multiple epochs.

        :param X: np.ndarray
        :param y: np.ndarray
        :param epochs: int
        """
        for epoch in range(epochs):
            loss = self.train_step(X, y)

            # TODO: maybe print or log loss
            ...