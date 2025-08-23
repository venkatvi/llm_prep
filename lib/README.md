# Core Library

Reusable ML components for experiments with comprehensive type annotations and enhanced training capabilities.

## Modules

- **`configs.py`** - Configuration dataclasses with type annotations (ExperimentConfig, TrainConfig, etc.)
- **`activations.py`** - Activation function factory with type safety (ReLU, Tanh, Sigmoid, etc.)
- **`loss_functions.py`** - Loss function factory with type hints (MSE, CrossEntropy, HuberLoss)
- **`train.py`** - Enhanced training utilities with encoder-decoder support, optimizers, schedulers
- **`logger.py`** - TensorBoard logging wrapper with type annotations
- **`utils.py`** - Plotting and weight initialization utilities with type safety

## Usage

### Basic Training
```python
from lib.configs import TrainConfig
from lib.train import TrainContext, train_model
from lib.activations import get_activation_layer

config = TrainConfig(epochs=1000, optimizer="adam", lr=0.001)
context = TrainContext(config)
train_loss, val_loss = train_model(model, x_train, y_train, x_val, y_val, context)
```

### Encoder-Decoder Training
```python
from lib.train import train_encoder_decoder, predict_encoder_decoder, TrainContext
import torch

# Sequence-to-sequence training
source_sequences = torch.randn(1000, 16, 1)  # [batch, src_len, input_dim]
target_sequences = torch.randn(1000, 20, 1)  # [batch, tgt_len, input_dim]

train_loss, val_loss = train_encoder_decoder(
    model, source_sequences, target_sequences, train_context
)

# Autoregressive prediction for encoder-decoder
starting_token = torch.randn(32, 1, 1)  # [batch, 1, input_dim]
generated_sequence = predict_encoder_decoder(
    model, source_sequences[:32], starting_token, num_steps=10, 
    log_dir="logs", run_name="encoder_decoder_predict"
)
```

### Autoregressive Prediction
```python
from lib.train import ar_predict

# Generate tokens autoregressively
input_sequence = torch.randn(32, 10, 1)  # [batch, seq_len, input_dim]
generated_tokens = ar_predict(
    model, input_sequence, num_steps_override=15, 
    expanding_context=True, max_seq_len=128,
    log_dir="logs", run_name="autoregressive_predict"
)
```

## Key Features

### Enhanced Training Functions
- **`train_model()`** - Standard training loop for regression/classification with validation splits
- **`train_encoder_decoder()`** - Specialized training for sequence-to-sequence models with teacher forcing
- **`predict_encoder_decoder()`** - Autoregressive prediction for encoder-decoder architectures
- **`ar_predict()`** - Token-by-token generation with expanding or sliding window context

### Type Annotations
All functions and classes include comprehensive type hints:
- Function parameters and return types
- Generic type parameters for flexibility
- Optional type handling with proper defaults
- Import organization following PEP8 standards

### Training Context Management
- **`TrainContext`** dataclass for centralized training configuration
- Optimizer and scheduler integration
- TensorBoard logging with structured naming
- Validation splitting and batch processing

### Data Utilities
- Automatic train/validation splits with shuffling
- DataLoader integration for memory-efficient training
- Support for various tensor shapes and batch sizes
- Random seed management for reproducibility