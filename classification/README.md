# CIFAR-10 Image Classification

PyTorch CNN for CIFAR-10 dataset (32×32 RGB images, 10 classes) with comprehensive type annotations and enhanced code quality.

## Features

- **CNN Architecture**: Optimized for CIFAR-10 (32×32 images) with type-annotated layers
- **Dataset**: Automated CIFAR-10 loading and preprocessing with type safety
- **Training**: Complete pipeline with validation and accuracy tracking
- **Logging**: TensorBoard integration with structured metrics
- **Type Safety**: Comprehensive type annotations following PEP 484/585 standards
- **Code Quality**: Enhanced documentation and error handling

## Quick Start

```bash
cd classification
python main.py  # Downloads CIFAR-10 automatically
```

## Files

- **`main.py`** - Entry point for classification with type annotations
- **`experiment.py`** - CIFARExperiment orchestrator with enhanced type safety
- **`cifar_cnn.py`** - CNN architecture with comprehensive type hints
- **`dataset.py`** - CIFAR-10 utilities with type-safe data loading
- **`configs.py`** - Model configuration with typed dataclasses

## CNN Architecture

```
Input: [batch, 3, 32, 32]
├── Conv Block 1: 3→32→32 channels, MaxPool → [batch, 32, 16, 16]
├── Conv Block 2: 32→64→64 channels, MaxPool → [batch, 64, 8, 8]
├── Flatten → [batch, 4096]
├── Linear: 4096→512 + ReLU
└── Linear: 512→10 (classes)
```

## Usage

```python
from typing import Optional
from lib.configs import ExperimentConfig, TrainConfig, DataConfig
from configs import CIFARModelConfig
from experiment import CIFARExperiment

# Type-annotated configuration
config: ExperimentConfig = ExperimentConfig(
    type="classification",
    train_config=TrainConfig(epochs=10, custom_loss="crossentropy", lr=0.001),
    data=DataConfig(use_dataloader=True, training_batch_size=64),
    model=CIFARModelConfig(input_channels=3)
)

# Type-safe experiment execution
experiment: CIFARExperiment = CIFARExperiment(config)
train_loss: float = experiment.train()
predictions: Optional[torch.Tensor] = experiment.predict()
```

## Code Quality Improvements

### Type Annotations
- All classes and functions include comprehensive type hints
- Generic type parameters for flexible tensor operations
- Proper handling of Optional types and Union types
- Import statements organized following PEP8 standards

### Enhanced Documentation
- Docstrings for all public methods and classes
- Clear parameter descriptions and return types
- Usage examples with type information
- Error handling documentation

## Expected Results

- **Training**: 60-70% accuracy achievable
- **Validation**: 55-65% on test set
- **Convergence**: 10-20 epochs typically
- **Parameters**: ~1.2M total