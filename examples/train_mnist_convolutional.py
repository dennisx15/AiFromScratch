from nn.backend import set_device, xp
from nn.layers.flatten import Flatten
from nn.model import Model
from nn.layers.dense import Dense
from nn.layers.conv2d import Conv2D
from nn.layers.activations import ReLU
from nn.losses.cross_entropy import CrossEntropyLoss
from nn.optimizers.adam import Adam
from nn.trainers.trainer import Trainer
from nn.data.visualizer import *

from sklearn.datasets import fetch_openml

"""
Train a simple MLP on MNIST using this framework.

Runs on CPU by default.
If you have an nvidia gpu, you can install cupy and set device to "gpu"
"""

def accuracy(model, X, y):
    logits = model.forward(X)
    preds = xp.argmax(logits, axis=1)
    return (preds == y).mean()


def load_data():
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff"
    )

    X = X.astype(float) / 255.0
    X = X.reshape(-1, 1, 28, 28)
    y = y.astype(int)

    return X, y


def main():
    # --------------------
    # Device
    # --------------------
    set_device("cpu")  # or "gpu" if you have an nvidia gpu

    # --------------------
    # Data
    # --------------------
    X, y = load_data()

    X = xp.array(X)
    y = xp.array(y)

    X_train, X_test = X[:100], X[100:200]
    y_train, y_test = y[:100], y[100:200]

    # --------------------
    # Model
    # --------------------
    model =Model([
    Conv2D(out_channels=5),
    ReLU(),
    Conv2D(out_channels=2, in_channels=5),
    ReLU(),
    Flatten(),
    Dense(1152, 32),
    ReLU(),
    Dense(32, 32),
    ReLU(),
    Dense(32, 10)
])

    # --------------------
    # Training setup
    # --------------------
    loss_fn = CrossEntropyLoss()
    optimizer = Adam()

    trainer = Trainer(model, loss_fn, optimizer)

    # --------------------
    # Train
    # --------------------
    losses, accuracies = trainer.fit(X_train, y_train, epochs=20, batch_size=10)

    # --------------------
    # Evaluate
    # --------------------
    print("Train acc:", accuracy(model, X_train, y_train))
    print("Test acc:", accuracy(model, X_test, y_test))

    plot_training(losses, accuracies)
    plot_confusion_matrix(model, X_test, y_test)




if __name__ == "__main__":
    main()