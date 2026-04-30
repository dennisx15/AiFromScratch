import matplotlib.pyplot as plt
from nn.backend import xp
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training(losses, accuracies):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title("Accuracy")

    plt.show()

def plot_confusion_matrix(model, X, y):
    """
    :param model: to make the predictions
    :param X: the inputs of a dataset
    :param y: the truth labels
    Plot the confusion matrix
    """

    logits = model.forward(X)
    preds = xp.argmax(logits, axis=1)
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def show_predictions(model, X, y, n=5):
    """
    :param model: to make the predictions
    :param X: the inputs of a dataset
    :param y: the truth labels
    :param n: number of random samples to choose from the dataset
    visualize the predictions
    """
    for i in range(n):
        img = X[i].reshape(28, 28)

        logits = model.forward(X[i:i+1])
        pred = logits.argmax(axis=1)[0]

        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {pred}, True: {y[i]}")
        plt.show()