# CIFAR-10 Image Classification

PyTorch CNN for CIFAR-10 dataset (32×32 RGB images, 10 classes).

## Features

- **CNN Architecture**: Optimized for CIFAR-10 (32×32 images)
- **Dataset**: Automated CIFAR-10 loading and preprocessing
- **Training**: Complete pipeline with validation and accuracy tracking
- **Logging**: TensorBoard integration

## Quick Start

```bash
cd classification
python main.py  # Downloads CIFAR-10 automatically
```

## Files

- **`main.py`** - Entry point for classification
- **`experiment.py`** - CIFARExperiment orchestrator
- **`cifar_cnn.py`** - CNN architecture
- **`dataset.py`** - CIFAR-10 utilities
- **`configs.py`** - Model configuration

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
from lib.configs import ExperimentConfig, TrainConfig, DataConfig
from configs import CIFARModelConfig
from experiment import CIFARExperiment

config = ExperimentConfig(
    type="classification",
    train_config=TrainConfig(epochs=10, custom_loss="crossentropy", lr=0.001),
    data=DataConfig(use_dataloader=True, training_batch_size=64),
    model=CIFARModelConfig(input_channels=3)
)

experiment = CIFARExperiment(config)
experiment.train()
```

## Expected Results

- **Training**: 60-70% accuracy achievable
- **Validation**: 55-65% on test set
- **Convergence**: 10-20 epochs typically
- **Parameters**: ~1.2M total