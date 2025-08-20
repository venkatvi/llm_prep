# Core Library

Reusable ML components for experiments.

## Modules

- **`configs.py`** - Configuration dataclasses (ExperimentConfig, TrainConfig, etc.)
- **`activations.py`** - Activation function factory (ReLU, Tanh, Sigmoid, etc.)
- **`loss_functions.py`** - Loss function factory (MSE, CrossEntropy, HuberLoss)
- **`train.py`** - Training utilities, optimizers, schedulers
- **`logger.py`** - TensorBoard logging wrapper
- **`utils.py`** - Plotting and weight initialization

## Usage

```python
from lib.configs import TrainConfig
from lib.train import TrainContext, train_model
from lib.activations import get_activation_layer

config = TrainConfig(epochs=1000, optimizer="adam", lr=0.001)
context = TrainContext(config)
train_loss, val_loss = train_model(model, x_train, y_train, x_val, y_val, context)
```