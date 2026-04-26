from nn.backend import set_device, xp
from nn.model import Model
from nn.layers.dense import Dense
from nn.layers.activations import ReLU
from nn.losses.cross_entropy import CrossEntropyLoss
from nn.optimizers.adam import Adam
from nn.trainers.trainer import Trainer

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

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # --------------------
    # Model
    # --------------------
    model = Model([
        Dense(784, 256),
        ReLU(),
        Dense(256, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10)
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
    trainer.fit(X_train, y_train, epochs=20, batch_size=64)

    # --------------------
    # Evaluate
    # --------------------
    print("Train acc:", accuracy(model, X_train, y_train))
    print("Test acc:", accuracy(model, X_test, y_test))

    # --------------------
    # Save model
    # --------------------


if __name__ == "__main__":
    main()