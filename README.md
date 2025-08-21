# PyTorch Machine Learning Framework

Comprehensive ML framework with regression, classification, transformer models, and custom autograd implementations.

## Structure

- `regression/` - Linear, non-linear, and transformer regression with experiment management
- `classification/` - CIFAR-10 CNN classification
- `autograd/` - Custom PyTorch autograd implementations (educational)
- `lib/` - Core library components (configs, training, logging, utils)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Regression
cd regression && python main.py --type nlinear --epochs 1000

# Transformer regression
cd regression && python main.py --type transformer --epochs 1000

# Classification  
cd classification && python main.py

# Custom autograd
cd autograd && python main.py
```

## Features

- **Models**: Linear regression, MLP, Transformer, CNN for CIFAR-10
- **Training**: Complete pipelines with validation, optimizers, schedulers
- **Experiment Management**: Structured configs, hyperparameter sweeps
- **Logging**: TensorBoard integration
- **Custom Autograd**: Educational gradient computation implementations

## Usage Examples

### Regression
```python
from lib.configs import ExperimentConfig, TrainConfig
from regression.experiment import RegressionExperiment

config = ExperimentConfig(
    type='nlinear',
    train_config=TrainConfig(epochs=1000, optimizer='adam', lr=0.001)
)
experiment = RegressionExperiment(config)
experiment.train()
```

### Classification
```python
from lib.configs import ExperimentConfig, TrainConfig, DataConfig
from classification.experiment import CIFARExperiment

config = ExperimentConfig(
    type="classification",
    train_config=TrainConfig(epochs=10, custom_loss="crossentropy"),
    data=DataConfig(use_dataloader=True, training_batch_size=64)
)
experiment = CIFARExperiment(config)
experiment.train()
```

### Custom Autograd
```python
import torch
from autograd.linear import Linear
from autograd.activations import ReLU

x = torch.randn(2, 3, requires_grad=True)
w = torch.randn(4, 3, requires_grad=True)
b = torch.randn(2, 4, requires_grad=True)

y = Linear.apply(x, w, b)
z = ReLU.apply(y)
loss = z.sum()
loss.backward()
```

## Documentation

- [`regression/README.md`](regression/README.md)
- [`classification/README.md`](classification/README.md)  
- [`autograd/README.md`](autograd/README.md)
- [`lib/README.md`](lib/README.md)
- [`transformer/`](transformer/) - Transformer encoder components