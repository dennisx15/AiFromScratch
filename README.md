# AI From Scratch

A modular neural network framework built from first principles using NumPy (and optional CuPy for GPU acceleration).
Designed to deeply understand backpropagation, optimization, and neural network internals — without relying on high-level libraries like PyTorch or TensorFlow.

---

## Features

* Fully connected (Dense) layers
* Activation functions (ReLU, Sigmoid, etc.)
* Loss functions (MSE, CrossEntropy)
* Optimizers (SGD, Adam)
* Modular architecture (layers, models, training loop)
* CPU (NumPy) and GPU (CuPy) support
* Built-in training pipeline

---

## Project Structure

```
AiFromScratch/
│
├── nn/                 # Core neural network library
│   ├── backend/        # NumPy / CuPy abstraction
│   ├── layers/         # Dense + activations
│   ├── losses/         # Loss functions
│   ├── models/         # Model architectures
│   ├── optimizers/     # SGD, Adam
│   ├── trainers/       # Training loop
│
├── examples/           # Example scripts
│   └── train_mnist.py
│
├── tests/              # Unit tests (pytest)
├── diagrams/           # Architecture diagrams
```

---

## Example Usage

```python
from nn.model import Model
from nn.trainers import trainer
from nn.layers.dense import Dense
from nn.layers.activations import ReLU
from nn.trainers.trainer import Trainer

model = Model([
    Dense(784, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, 10)
    ])

trainer = Trainer(model)
trainer.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

## Results

Example (MNIST):

* Accuracy: ~95%+ (CPU)
* Training time: ~30–60 seconds
* Model: 2-layer MLP

---

##  How It Works

* Forward pass computes outputs layer-by-layer
* Backward pass computes gradients using the chain rule
* Optimizers update parameters using computed gradients
* Training loop handles batching, loss calculation, and updates

---

## Goals

* Understand neural networks at a low level
* Implement backpropagation from scratch
* Build a scalable and modular ML framework
* Bridge theory and real-world implementation

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running an Example

```bash
cd examples
python train_mnist.py
```

---

## Status

## Running an example model
navigate to examples and run train_mnist.py
