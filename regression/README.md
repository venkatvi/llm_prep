# Regression Models

A comprehensive PyTorch-based regression framework featuring linear and non-linear models with advanced training utilities, experiment management, and professional documentation.

## Features

- **Model Types**: Linear regression and Multi-Layer Perceptron (MLP) 
- **Architecture Options**: Configurable layers, dimensions, activation functions, and residual connections
- **Training Pipeline**: Complete training loops with validation, multiple optimizers, and learning rate scheduling
- **Experiment Management**: Structured configuration system and automated hyperparameter sweeps
- **Data Processing**: Synthetic data generation, DataLoader support, and reproducible experiments
- **Visualization**: TensorBoard integration and scatter plot visualization
- **Professional Documentation**: Complete docstrings, type annotations, and usage examples

## Quick Start

```bash
# Install dependencies
pip install torch matplotlib numpy pandas tensorboard

# Linear regression with default settings
python main.py --type linear --epochs 1000 --lr 0.01

# Non-linear regression with MLP
python main.py --type nlinear --epochs 1000 --lr 0.001 --latent_dims "128,64,32"

# Hyperparameter sweep
python experiment_sweep.py
```

## Project Structure

### Application Layer
- **`main.py`** - CLI entry point with comprehensive argument parsing
- **`experiment.py`** - RegressionExperiment class orchestrating the complete ML pipeline
- **`experiment_sweep.py`** - Hyperparameter grid search with cross product generation
- **`dataset.py`** - PyTorch Dataset and DataLoader utilities for batch processing
- **`e_linear_reg.py`** - LinearRegressionModel with optional activation functions
- **`e_non_linear_reg.py`** - MLP with configurable architecture and residual connections

### Core Library Dependencies
Uses shared components from `../lib/` for configuration, training, logging, and utilities. See [`../lib/README.md`](../lib/README.md) for details.

## Usage Examples

### Basic Experiment
```python
from lib.configs import ExperimentConfig, TrainConfig, DataConfig, ModelConfig
from experiment import RegressionExperiment

config = ExperimentConfig(
    type="nlinear",
    name="demo_experiment",
    train_config=TrainConfig(epochs=1000, optimizer="adam", lr=0.001, lr_scheduler="steplr", step_size=10),
    data=DataConfig(use_dataloader=True, training_batch_size=32),
    model=ModelConfig(custom_act="relu", num_latent_layers=3, latent_dims=[128, 64, 32])
)

experiment = RegressionExperiment(config)
experiment.train()
predictions = experiment.predict()
experiment.plot_results(predictions)
```

### Hyperparameter Sweeps
```bash
python experiment_sweep.py  # Runs automated grid search
```

Generates cross products of parameter combinations:
- **Optimizers**: SGD, Adam, RMSprop
- **Learning rates**: 0.01, 0.001, 0.0001
- **Activations**: ReLU, Tanh, GELU, LeakyReLU, SiLU
- **Architectures**: Various layer configurations
- **Loss functions**: MSE, Huber Loss

## Model Architectures

### Linear Regression
- **Architecture**: Single linear layer (1 → 1) with optional activation
- **Data**: y = 100x + noise (100 samples)
- **Use case**: Simple linear relationships

### Non-Linear Regression (MLP)
- **Architecture**: Configurable multi-layer perceptron
- **Features**: Variable layers, dimensions, activation functions, residual connections
- **Data**: y = 4x² + 2x + noise (100 samples)
- **Use case**: Complex non-linear relationships

Example MLP with `--latent_dims "128,64,32"`:
```
Input(1) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(32) → ReLU → Output(1)
```

## Configuration Options

### Command Line Arguments
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--type` | `linear`, `nlinear` | `linear` | Model type |
| `--epochs` | integer | `1000` | Training epochs |
| `--lr` | float | `0.01` | Learning rate |
| `--optimizer` | `adam`, `sgd`, `rmsprop` | `adam` | Optimizer |
| `--lr_scheduler` | `steplr`, `exp`, `reduceonplat`, `cosine` | `reduceonplat` | Learning rate scheduler |
| `--custom_loss` | `mse`, `huber` | `mse` | Loss function |
| `--custom_act` | `relu`, `tanh`, `gelu`, etc. | `relu` | Activation function |
| `--latent_dims` | comma-separated ints | `[256]` | Hidden layer dimensions |
| `--use_dataloader` | flag | `False` | Enable DataLoader batching |
| `--training_batch_size` | integer | `8` | Batch size |

### Advanced Options
```bash
# Complex MLP with residual connections
python main.py \
  --type nlinear \
  --latent_dims "512,256,128" \
  --num_latent_layers 3 \
  --custom_act gelu \
  --allow_residual \
  --use_dataloader \
  --training_batch_size 16 \
  --custom_loss huber \
  --epochs 2000 \
  --lr_scheduler steplr
```

## Training Features

- **Data Splitting**: Automatic 80/20 train/validation split
- **Loss Functions**: MSE (standard), Huber Loss (robust to outliers)
- **Optimizers**: Adam, SGD, RMSprop with configurable learning rates
- **Schedulers**: StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealing
- **Initialization**: Kaiming uniform weight initialization
- **Logging**: TensorBoard integration with loss and learning rate tracking

## TensorBoard Monitoring

```bash
# Start training (logs created automatically)
python main.py --type nlinear --epochs 1000

# View in TensorBoard
tensorboard --logdir=./logs
# Open http://localhost:6006
```

**Logged Metrics:**
- Training/validation loss per epoch
- Learning rate changes
- Model predictions vs targets visualization

## Experiment Framework

### Complete Pipeline
```python
# The RegressionExperiment class provides:
experiment = RegressionExperiment(config)

# 1. Automatic model instantiation based on config
# 2. Synthetic data generation with optional fixed seeds
# 3. Training context setup (optimizer, scheduler, loss)
# 4. Training execution with progress logging
# 5. Prediction and performance evaluation
# 6. Results visualization and TensorBoard logging
```

### Hyperparameter Grid Search
The `experiment_sweep.py` module provides automated parameter space exploration:
- **Cross Products**: All parameter combinations tested systematically
- **Error Handling**: Failed experiments don't stop the sweep
- **Progress Tracking**: Clear status reporting for long-running sweeps
- **Reproducible**: Fixed random seeds for consistent comparisons

## Expected Results

### Training Progress
```
Epoch 10/1000, Train Loss: 2.4531, Val Loss: 2.4891, LR: 0.010000
Epoch 100/1000, Train Loss: 1.8234, Val Loss: 1.8567, LR: 0.008000
Epoch 1000/1000, Train Loss: 0.1234, Val Loss: 0.1456, LR: 0.001000
```

### Model Performance
- **Linear**: Typically achieves low loss on linear synthetic data
- **MLP**: Capable of learning complex non-linear relationships
- **Convergence**: Usually converges within 500-1000 epochs
- **Validation**: Close train/validation performance indicates good generalization

## Dependencies

```bash
pip install torch matplotlib numpy pandas tensorboard
```

## Advanced Usage

See the comprehensive examples in the code and experiment sweep configurations for advanced usage patterns including custom architectures, loss functions, and training strategies.