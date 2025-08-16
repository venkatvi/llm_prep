# Regression Models

Linear and non-linear regression implementations using PyTorch with comprehensive training utilities.

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

## Files

- `main.py` - Entry point with comprehensive CLI interface
- `e_linear_reg.py` - Linear regression model with activation support
- `e_non_linear_reg.py` - MLP model with flexible architecture
- `train.py` - Training utilities and optimization functions
- `dataset.py` - PyTorch Dataset and DataLoader utilities
- `activations.py` - Activation function factory
- `loss_functions.py` - Custom loss functions and factory
- `logger.py` - TensorBoard logging utilities
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
# Complex MLP with multiple layers, custom activation, and DataLoader
python main.py \
  --type non-linear \
  --latent_dims "512,256,128" \
  --num_latent_layers 3 \
  --custom_act gelu \
  --allow_residual \
  --use_dataloader \
  --training_batch_size 16 \
  --custom_loss huber \
  --epochs 2000 \
  --lr 0.001

# Linear model with custom activation
python main.py \
  --type linear \
  --custom_act tanh \
  --epochs 1000
```

### Available Options

- `--type`: `linear` or `non-linear` (default: linear)
- `--epochs`: Number of training epochs (default: 1000)
- `--lr`: Learning rate (default: 0.01)
- `--latent_dims`: Comma-separated layer dimensions (e.g., "512,256,128")
- `--num_latent_layers`: Number of hidden layers (default: 1)
- `--custom_act`: Activation function - `relu`, `tanh`, `sigmoid`, `leakyrelu`, `gelu`, `silu` (default: relu)
- `--custom_loss`: Loss function - `mse`, `huber`, `crossentropy` (default: mse)
- `--allow_residual`: Enable residual connections (flag)
- `--use_dataloader`: Use DataLoader for batch processing (flag)
- `--training_batch_size`: Batch size for DataLoader (default: 8)
- `--optimizer`: `adam`, `sgd`, or `rmsprop` (default: adam)
- `--lr_scheduler`: `steplr`, `exp`, `reduceonplat`, or `cosine` (default: reduceonplat)

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