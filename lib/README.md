# Core Library (`lib/`)

This directory contains the core library components that provide reusable functionality for machine learning experiments. These modules are designed to be imported and used by application-level code in the regression framework.

## Library Components

### Configuration Management
- **`configs.py`** - Structured configuration dataclasses for all experiment parameters
  - `ExperimentConfig`: Top-level experiment configuration
  - `TrainConfig`: Training loop parameters (epochs, loss, optimizer, learning rate)
  - `DataConfig`: Data processing and loading configuration
  - `ModelConfig`: Neural network architecture configuration

### Model Components
- **`activations.py`** - Activation function factory supporting 6 different activation types
  - Supports: ReLU, Tanh, Sigmoid, LeakyReLU, GELU, SiLU
  - Factory pattern for easy activation selection
- **`loss_functions.py`** - Custom loss functions and loss function factory
  - Standard losses: MSE, CrossEntropy
  - Custom implementations: HuberLoss for robust regression
  - Factory pattern for loss function selection

### Training Infrastructure
- **`train.py`** - Core training utilities and context management
  - `TrainContext`: Training context with optimizer, scheduler, and loss configuration
  - Training loops: `train_model()`, `train_model_with_dataloader()`
  - Optimizer factory: Adam, SGD, RMSprop
  - Scheduler factory: StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
  - Data utilities: `split_data()`, `predict_model()`

### Logging and Visualization
- **`logger.py`** - TensorBoard logging wrapper with comprehensive metric tracking
  - `Logger` class for structured logging
  - Supports scalar metrics, tensor logging, and figure visualization
  - Automatic experiment organization and cleanup
- **`utils.py`** - Visualization utilities and neural network initialization
  - `plot_results()`: Scatter plot visualization with TensorBoard integration
  - `init_weights()`: Kaiming uniform weight initialization for better convergence

## Design Principles

### Modularity
Each module has a single responsibility and can be used independently:
```python
from lib.configs import ExperimentConfig, TrainConfig
from lib.activations import get_activation_layer
from lib.train import TrainContext, train_model
from lib.logger import Logger
```

### Configuration-Driven
All components are designed to work with structured configurations:
```python
# Type-safe configuration
config = TrainConfig(
    epochs=1000,
    optimizer="adam",
    lr=0.001,
    custom_loss="mse"
)

# Automatic factory instantiation
context = TrainContext(config)
```

### Professional Documentation
Every function and class includes comprehensive docstrings with:
- Clear parameter specifications with types
- Return value documentation
- Usage examples
- Implementation notes
- Error handling documentation

## Usage Patterns

### Basic Training Setup
```python
from lib.configs import TrainConfig, ModelConfig
from lib.train import TrainContext, train_model
from lib.activations import get_activation_layer
from lib.loss_functions import get_loss_function

# Configure training
train_config = TrainConfig(epochs=1000, optimizer="adam", lr=0.001)
model_config = ModelConfig(custom_act="relu")

# Set up training context
context = TrainContext(train_config)
activation = get_activation_layer(model_config.custom_act)
loss_fn = get_loss_function(train_config.custom_loss)

# Execute training
train_loss, val_loss = train_model(model, x_train, y_train, x_val, y_val, context)
```

### Logging and Visualization
```python
from lib.logger import Logger
from lib.utils import plot_results

# Set up logging
logger = Logger(log_dir="./logs", run_name="experiment_1")

# Log metrics during training
logger.log_scalar("train_loss", loss_value, epoch)
logger.log_scalar("learning_rate", lr, epoch)

# Create and log visualizations
plot_results(inputs, targets, predictions, "./logs", "experiment_1")
logger.close()
```

### Factory Pattern Usage
```python
from lib.activations import get_activation_layer
from lib.loss_functions import get_loss_function

# Dynamic component selection
activation_names = ["relu", "gelu", "tanh"]
loss_names = ["mse", "huber"]

for act_name in activation_names:
    activation = get_activation_layer(act_name)
    for loss_name in loss_names:
        loss_fn = get_loss_function(loss_name)
        # Use in experiments...
```

## Integration with Application Layer

The lib components are designed to be consumed by application-level code:

- **Experiment orchestration** imports these modules to build complete ML pipelines
- **Model implementations** use activations and loss functions
- **Training scripts** use training utilities and context management
- **Visualization tools** use logging and plotting utilities

## Dependencies

The library requires:
- PyTorch 2.1.0+ (core ML framework)
- TensorBoard 2.20.0+ (logging and visualization)
- matplotlib 3.7.2+ (plotting utilities)
- numpy 1.24.3+ (numerical operations)

## Testing and Quality

All library components include:
- Comprehensive type annotations
- Input validation and error handling
- Professional documentation standards
- Modular design for easy testing
- Configuration-driven behavior for reproducibility