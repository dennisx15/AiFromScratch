from nn.backend import xp

class Trainer:
    """
    Handles the training process of a model.

    Responsible for coordinating forward pass, loss computation,
    backward pass, and parameter updates.
    """

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, X, y):
        """
        Perform one training step (forward + backward + update).
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

    def accuracy(self, X, y):
        logits = self.model.forward(X)
        preds = xp.argmax(logits, axis=1)
        return (preds == y).mean()

    def fit(self, X, y, epochs, batch_size=None):
        """
        Train the model for multiple epochs.

        :param X: np.ndarray
        :param y: np.ndarray
        :param epochs: int
        :param batch_size: int or None
            If None, use full batch.
        :return: (losses, accuracies)
        """

        losses = []
        accuracies = []
        # default: full batch
        if batch_size is None:
            batch_size = X.shape[0]

        for epoch in range(epochs):

            #Shuffle data
            indices = xp.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            num_batches = 0

            #Loop over batches
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                loss = self.train_step(X_batch, y_batch)

                epoch_loss += loss
                num_batches += 1

            #Average loss across batches
            epoch_loss /= num_batches
            losses.append(epoch_loss)

            #Logging
            acc = self.accuracy(X, y)
            accuracies.append(acc)

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: loss = {epoch_loss:.4f}, acc = {acc:.4f}")

        return losses, accuracies