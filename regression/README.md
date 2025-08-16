# Regression Models

A comprehensive PyTorch-based regression framework featuring linear and non-linear models with advanced training utilities, TensorBoard integration, and extensive documentation. This project provides a complete machine learning pipeline with configurable architectures, multiple optimizers, and professional-grade logging capabilities.

## Features

- **Linear Regression**: Simple linear model (y = ax + b) with optional activation functions
- **Non-Linear Regression**: Multi-Layer Perceptron (MLP) with configurable architecture
- **Flexible Architecture**: Variable number of layers and dimensions via comma-separated values
- **Activation Functions**: ReLU, Tanh, Sigmoid, LeakyReLU, GELU, SiLU support
- **Loss Functions**: MSE, Huber Loss, CrossEntropy options
- **Residual Connections**: Optional skip connections in MLP layers
- **Training Pipeline**: Complete training loop with validation splits
- **TensorBoard Integration**: Real-time logging and visualization of training metrics
- **DataLoader Support**: Batch processing with PyTorch DataLoaders
- **Optimizer Support**: Adam, SGD, RMSprop optimizers
- **Learning Rate Scheduling**: StepLR, Exponential, ReduceLROnPlateau, Cosine Annealing
- **Data Utilities**: Automatic shuffling and train/validation splits
- **Weight Initialization**: Kaiming uniform initialization for better convergence
- **Experiment Management**: Structured configuration system with dataclasses
- **Hyperparameter Sweeps**: Automated grid search across parameter combinations
- **Reproducible Experiments**: Fixed random seeds and comprehensive logging

## Project Structure

All files include comprehensive documentation with detailed docstrings, parameter specifications, usage examples, and implementation details.

### Core Framework
- **`main.py`** - Entry point with comprehensive CLI interface and argument parsing
- **`experiment.py`** - Experiment orchestrator class managing the complete ML pipeline
- **`configs.py`** - Structured configuration dataclasses for all experiment parameters
- **`experiment_sweep.py`** - Hyperparameter grid search utilities with cross product generation

### Model Architecture
- **`e_linear_reg.py`** - Linear regression model (LinearRegressionModel) with optional activation functions
- **`e_non_linear_reg.py`** - Multi-layer perceptron (MLP) with configurable architecture and residual connections
- **`activations.py`** - Activation function factory supporting 6 different activation types
- **`loss_functions.py`** - Custom loss functions including HuberLoss and loss function factory

### Training Infrastructure
- **`train.py`** - Training utilities including TrainContext dataclass, train/validation loops, and optimizer factories
- **`dataset.py`** - PyTorch Dataset (RegressionDataset) and DataLoader utilities for batch processing
- **`logger.py`** - TensorBoard logging wrapper (Logger class) with scalars, tensors, and figure logging
- **`utils.py`** - Visualization utilities (plot_results) and weight initialization (init_weights)

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Linear regression with default settings
python main.py --type linear --epochs 1000 --lr 0.01

# Non-linear regression with single hidden layer
python main.py --type nlinear --epochs 1000 --lr 0.001 --latent_dims "256"
```

### Experiment Framework

The project includes a powerful experiment management system using structured configurations:

```python
from configs import ExperimentConfig, TrainConfig, DataConfig, ModelConfig
from experiment import Experiment

# Create structured experiment configuration
config = ExperimentConfig(
    type="nlinear",
    name="my_experiment",
    train_config=TrainConfig(
        epochs=1000,
        custom_loss="mse",
        optimizer="adam",
        lr=0.001,
        lr_scheduler="reduceonplat"
    ),
    data=DataConfig(
        use_dataloader=True,
        training_batch_size=32,
        fix_random_seed=True
    ),
    model=ModelConfig(
        custom_act="relu",
        num_latent_layers=3,
        latent_dims=[128, 64, 32],
        allow_residual=True
    )
)

# Run complete experiment pipeline
experiment = Experiment(config)
experiment.train()
predictions = experiment.predict()
experiment.plot_results(predictions)
```

### Hyperparameter Sweeps

Automated grid search across parameter combinations:

```bash
# Run hyperparameter sweep with predefined parameter arrays
python experiment_sweep.py

# This generates cross products of:
# - Optimizers: SGD, Adam, RMSprop
# - Learning rates: 0.01, 0.001, 0.0001
# - Activations: ReLU, Tanh, GELU, LeakyReLU, SiLU
# - Model architectures: Various layer configurations
# - And many more parameter combinations
```

### Advanced Options

```bash
# Complex MLP with multiple layers, custom activation, and DataLoader
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
  --lr 0.001 \
  --run_name "complex_mlp_experiment"

# Linear model with custom activation and named run
python main.py \
  --type linear \
  --custom_act tanh \
  --epochs 1000 \
  --run_name "linear_tanh_baseline"
```

### Available Options

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--type` | `linear`, `nlinear` | `linear` | Type of regression model |
| `--epochs` | integer | `1000` | Number of training epochs |
| `--lr` | float | `0.01` | Learning rate |
| `--latent_dims` | comma-separated ints | `[256]` | Hidden layer dimensions (e.g., "512,256,128") |
| `--num_latent_layers` | integer | `1` | Number of hidden layers |
| `--custom_act` | `relu`, `tanh`, `sigmoid`, `leakyrelu`, `gelu`, `silu` | `relu` | Activation function |
| `--custom_loss` | `mse`, `huber`, `crossentropy` | `mse` | Loss function |
| `--allow_residual` | flag | `False` | Enable residual connections |
| `--use_dataloader` | flag | `False` | Use DataLoader for batch processing |
| `--training_batch_size` | integer | `8` | Batch size for DataLoader |
| `--optimizer` | `adam`, `sgd`, `rmsprop` | `adam` | Optimizer type |
| `--lr_scheduler` | `steplr`, `exp`, `reduceonplat`, `cosine` | `reduceonplat` | Learning rate scheduler |
| `--run_name` | string | auto-generated | Name for TensorBoard run |

## Data Generation

- **Linear**: y = 100x + noise
- **Non-linear**: y = 4x² + 2x + noise

Both datasets use 100 samples with 80/20 train/validation split and automatic shuffling.

## Model Architecture

### Linear Model
- Single linear layer: 1 input → 1 output
- Optional activation function (ReLU, Tanh, Sigmoid, etc.)

### MLP Model
- Configurable architecture with multiple layers
- Variable layer dimensions via `--latent_dims`
- Customizable activation functions between layers
- Optional residual connections (skip connections)
- Final output layer: last_hidden_dim → 1 output

**Example MLP with `--latent_dims "512,256,128"`:**
```
Input (1) → Linear(512) → Activation → Linear(256) → Activation → Linear(128) → Activation → Output(1)
```

## Training Features

- **Automatic data splitting**: 80% train, 20% validation
- **Progress logging**: Loss and learning rate tracking
- **TensorBoard logging**: Real-time visualization of training metrics
- **Validation monitoring**: Prevents overfitting
- **Learning rate scheduling**: Adaptive learning rate adjustment

## TensorBoard Monitoring

The framework automatically logs training metrics to TensorBoard:

```bash
# Start training (logs are automatically created)
python main.py --type non-linear --epochs 1000

# View logs in TensorBoard (in a separate terminal)
tensorboard --logdir=./logs
# Open http://localhost:6006 in your browser
```

**Logged Metrics:**
- Training loss per epoch
- Validation loss per epoch
- Learning rate changes
- Model predictions vs targets

## Example Output

```
Epoch 10/1000, Train Loss: 2.4531, Val Loss: 2.4891, LR: 0.010000
Epoch 20/1000, Train Loss: 1.8234, Val Loss: 1.8567, LR: 0.010000
...
Target: 245.67, Actual: 243.12
Target: 189.34, Actual: 191.78
MSE: 2.1234
```

A matplotlib scatter plot will display showing:
- Red circles (o) for target values
- Blue stars (*) for predicted values

## Code Documentation

This project features comprehensive documentation with professional-grade docstrings throughout:

### Documentation Features

- **Class Documentation**: Detailed descriptions of purpose, architecture, and usage patterns
- **Method Documentation**: Complete parameter specifications with types, descriptions, and examples
- **Return Value Documentation**: Clear specifications of return types and meanings
- **Error Handling**: Documented exceptions and error conditions
- **Usage Examples**: Code examples demonstrating proper usage
- **Implementation Notes**: Design rationale and implementation details

### Key Documented Classes

#### Core Framework
- **`Experiment`**: Complete experiment orchestrator managing the ML pipeline
- **`ExperimentConfig`**: Top-level configuration combining all parameter groups
- **`TrainConfig`**: Training loop parameters (epochs, loss, optimizer, learning rate)
- **`DataConfig`**: Data processing and loading configuration
- **`ModelConfig`**: Neural network architecture configuration

#### Models and Training
- **`LinearRegressionModel`**: Linear model with optional activation functions
- **`MLP`**: Multi-layer perceptron with configurable architecture and residual connections
- **`TrainContext`**: Training context with optimizer, scheduler, and loss configuration
- **`RegressionDataset`**: PyTorch Dataset for CSV-based regression data
- **`HuberLoss`**: Custom robust loss function implementation
- **`Logger`**: TensorBoard logging wrapper with comprehensive metric tracking

#### Utility Functions
- **`generate_experiment_configurations()`**: Grid search configuration generator
- **`run_experiment_sweep()`**: Automated experiment execution manager
- **`get_activation_layer()`**: Activation function factory
- **`get_loss_function()`**: Loss function factory

All functions include detailed docstrings following Python documentation standards with clear parameter types, return specifications, and usage examples.

## Dependencies

```
torch==2.1.0
matplotlib==3.7.2
numpy==1.24.3
pandas
tensorboard==2.20.0
pillow
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```