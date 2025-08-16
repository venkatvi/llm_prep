# Regression Models

Linear and non-linear regression implementations using PyTorch with comprehensive training utilities.

## Features

- **Linear Regression**: Simple linear model (y = ax + b)
- **Non-Linear Regression**: Multi-Layer Perceptron (MLP) for complex relationships
- **Training Pipeline**: Complete training loop with validation splits
- **Optimizer Support**: Adam, SGD, RMSprop optimizers
- **Learning Rate Scheduling**: StepLR, Exponential, ReduceLROnPlateau, Cosine Annealing
- **Data Utilities**: Automatic shuffling and train/validation splits
- **Weight Initialization**: Kaiming uniform initialization for better convergence

## Files

- `main.py` - Entry point with CLI interface
- `e_linear_reg.py` - Linear regression model
- `e_non_linear_reg.py` - MLP model for non-linear regression
- `train.py` - Training utilities and optimization functions
- `utils.py` - Visualization and weight initialization utilities

## Usage

### Basic Training

```bash
# Linear regression
python main.py --type linear --epochs 1000 --lr 0.01

# Non-linear regression
python main.py --type non-linear --epochs 1000 --lr 0.001 --hidden_dim 256
```

### Advanced Options

```bash
# Custom optimizer and scheduler
python main.py \
  --type non-linear \
  --epochs 2000 \
  --lr 0.01 \
  --optimizer adam \
  --lr_scheduler reduceonplat \
  --hidden_dim 512
```

### Available Options

- `--type`: `linear` or `non-linear` (default: linear)
- `--epochs`: Number of training epochs (default: 1000)
- `--lr`: Learning rate (default: 0.01)
- `--hidden_dim`: Hidden layer size for MLP (default: 256)
- `--optimizer`: `adam`, `sgd`, or `rmsprop` (default: adam)
- `--lr_scheduler`: `steplr`, `exp`, `reduceonplat`, or `cosine` (default: reduceonplat)

## Data Generation

- **Linear**: y = 100x + noise
- **Non-linear**: y = 4x² + 2x + noise

Both datasets use 100 samples with 80/20 train/validation split and automatic shuffling.

## Model Architecture

### Linear Model
- Single linear layer: 1 input → 1 output
- No activation function

### MLP Model
- Input layer: 1 input → hidden_dim
- Hidden layer: ReLU activation
- Output layer: hidden_dim → 1 output

## Training Features

- **Automatic data splitting**: 80% train, 20% validation
- **Progress logging**: Loss and learning rate tracking
- **Validation monitoring**: Prevents overfitting
- **Learning rate scheduling**: Adaptive learning rate adjustment

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