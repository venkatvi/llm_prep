# CIFAR-10 Image Classification

A comprehensive PyTorch-based image classification framework for the CIFAR-10 dataset featuring convolutional neural networks, experiment management, and professional documentation.

## Features

- **CNN Architecture**: Purpose-built convolutional neural network optimized for CIFAR-10 (32×32 RGB images)
- **CIFAR-10 Dataset**: Automated loading, preprocessing, and batching of 60,000 labeled images across 10 classes
- **Experiment Management**: Structured configuration system leveraging the core library framework
- **Training Pipeline**: Complete training loop with validation evaluation and accuracy tracking
- **TensorBoard Logging**: Real-time visualization of training metrics, loss, and classification accuracy
- **Professional Documentation**: Complete docstrings, type annotations, and usage examples

## Quick Start

```bash
# Install dependencies
pip install torch torchvision tensorboard numpy

# Train CNN on CIFAR-10 (downloads dataset automatically on first run)
cd classification
python main.py
```

## Project Structure

### Application Layer
- **`main.py`** - Entry point for CIFAR-10 classification experiments
- **`experiment.py`** - CIFARExperiment class orchestrating the classification pipeline
- **`configs.py`** - CIFAR-specific model configuration (extends base ModelConfig)
- **`cifar_cnn.py`** - CNN architecture optimized for 32×32 CIFAR-10 images
- **`dataset.py`** - CIFAR-10 dataset loading and preprocessing utilities

### Core Library Dependencies
Uses shared components from `../lib/` for configuration, training, and logging. See [`../lib/README.md`](../lib/README.md) for details.

## CIFAR-10 Dataset

### Overview
- **Image Size**: 32×32 pixels, 3 RGB channels
- **Classes**: 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training Samples**: 50,000 images (5,000 per class)
- **Test Samples**: 10,000 images (1,000 per class)
- **Total Size**: ~170 MB

### Preprocessing
- **Normalization**: Pixel values normalized to [-1, 1] range
- **Batching**: Configurable batch size (default: 64)
- **Shuffling**: Automatic shuffling for training and validation
- **Transforms**: ToTensor + Normalize pipeline

## CNN Architecture

### CIFARCNN Model
```
Input: [batch_size, 3, 32, 32]
│
├── Conv Block 1
│   ├── Conv2d(3→32, 3×3) + ReLU     → [batch, 32, 32, 32]
│   ├── Conv2d(32→32, 3×3) + ReLU    → [batch, 32, 32, 32]
│   └── MaxPool2d(2×2)               → [batch, 32, 16, 16]
│
├── Conv Block 2
│   ├── Conv2d(32→64, 3×3) + ReLU    → [batch, 64, 16, 16]
│   ├── Conv2d(64→64, 3×3) + ReLU    → [batch, 64, 16, 16]
│   └── MaxPool2d(2×2)               → [batch, 64, 8, 8]
│
├── Classifier
│   ├── Flatten()                   → [batch, 4096]
│   ├── Linear(4096→512) + ReLU      → [batch, 512]
│   └── Linear(512→10)              → [batch, 10]
│
Output: [batch_size, 10] (class logits)
```

### Architecture Details
- **Parameters**: ~1.2M total parameters
- **Receptive Field**: Progressive expansion through convolutional layers
- **Feature Maps**: 32 → 64 feature maps across blocks
- **Spatial Reduction**: 32×32 → 16×16 → 8×8 via max pooling

## Usage Examples

### Basic Classification
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.configs import ExperimentConfig, TrainConfig, DataConfig
from configs import CIFARModelConfig
from experiment import CIFARExperiment

# Configure experiment
config = ExperimentConfig(
    type="classification",
    name="CIFAR10_Demo",
    train_config=TrainConfig(
        epochs=10,
        custom_loss="crossentropy",
        optimizer="adam",
        lr=0.001,
        lr_scheduler="steplr",
        step_size=5
    ),
    data=DataConfig(
        use_dataloader=True,
        training_batch_size=64,
        fix_random_seed=42
    ),
    model=CIFARModelConfig(input_channels=3)
)

# Run experiment
experiment = CIFARExperiment(config)
experiment.train()
predictions = experiment.predict()
print(f"Accuracy: {experiment.accuracy:.4f}")
```

### Advanced Configuration
```python
# High-performance setup
config = ExperimentConfig(
    type="classification",
    name="CIFAR10_Advanced",
    train_config=TrainConfig(
        epochs=50,
        custom_loss="crossentropy",
        optimizer="adam",
        lr=0.0001,                    # Lower learning rate
        lr_scheduler="reduceonplat",  # Adaptive scheduling
        step_size=10                  # Step size for scheduler
    ),
    data=DataConfig(
        use_dataloader=True,
        training_batch_size=128,      # Larger batches
        fix_random_seed=42
    ),
    model=CIFARModelConfig(input_channels=3)
)
```

## Training Features

### Training Pipeline
- **Loss Function**: CrossEntropy loss for multi-class classification
- **Optimizer**: Adam optimizer with configurable learning rate
- **Scheduler**: StepLR or ReduceLROnPlateau learning rate scheduling
- **Validation**: Automatic accuracy evaluation on test set each epoch
- **Logging**: Real-time TensorBoard logging of training metrics

### Data Processing
- **Automatic Loading**: Downloads CIFAR-10 on first run
- **Preprocessing**: Tensor conversion and normalization
- **Batching**: Efficient DataLoader with configurable batch sizes
- **Reproducibility**: Optional fixed random seeds

## TensorBoard Monitoring

```bash
# Start training (logs created automatically)
python main.py

# View logs in TensorBoard
tensorboard --logdir=./logs
# Open http://localhost:6006
```

### Logged Metrics
- **Training Loss**: CrossEntropy loss per epoch
- **Validation Loss**: Loss on test set per epoch  
- **Classification Accuracy**: Top-1 accuracy on validation set
- **Learning Rate**: Learning rate schedule tracking

### Experiment Organization
```
logs/
├── CIFAR10_Demo/
├── CIFAR10_Advanced/
└── CIFAR10_Baseline/
```

## Expected Performance

### Training Progress
```
Epoch 1/10, Train Loss: 2.154, Val Loss: 1.988, Accuracy: 0.285
Epoch 2/10, Train Loss: 1.823, Val Loss: 1.712, Accuracy: 0.372
Epoch 5/10, Train Loss: 1.346, Val Loss: 1.457, Accuracy: 0.483
Epoch 10/10, Train Loss: 0.988, Val Loss: 1.235, Accuracy: 0.565
```

### Typical Results
- **Convergence**: Usually converges within 10-20 epochs
- **Training Accuracy**: 60-70% achievable with basic configuration
- **Validation Accuracy**: 55-65% on test set
- **Training Time**: ~2-5 minutes per epoch on modern GPU

## Model Testing

### Direct Model Usage
```python
from cifar_cnn import CIFARCNN
import torch

# Create and test model
model = CIFARCNN(input_channels=3)
test_input = torch.randn(32, 3, 32, 32)  # Batch of 32 images

model.eval()
with torch.no_grad():
    logits = model(test_input)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Predictions shape: {predictions.shape}")  # [32]
print(f"Sample predictions: {predictions[:5]}")   # First 5 classes
```

## Configuration Options

### Model Configuration
```python
model_config = CIFARModelConfig(
    input_channels=3,        # RGB channels
    # Inherited from ModelConfig (not used in CNN):
    custom_act="relu",       
    num_latent_layers=0,     
    latent_dims=[],          
    allow_residual=False     
)
```

### Training Configuration
```python
train_config = TrainConfig(
    epochs=20,                    # Training epochs
    custom_loss="crossentropy",   # Multi-class loss
    optimizer="adam",             # Adam optimizer
    lr=0.001,                    # Learning rate
    lr_scheduler="steplr",       # Learning rate decay
    step_size=5                  # Steps for scheduler decay
)
```

### Data Configuration
```python
data_config = DataConfig(
    use_dataloader=True,          # Required for image data
    training_batch_size=64,       # Batch size
    fix_random_seed=42           # Reproducible results
)
```

## Integration with Core Library

### Shared Components
- **Configuration System**: Uses base ExperimentConfig from `lib.configs`
- **Training Infrastructure**: Leverages `lib.train` for DataLoader training
- **Logging System**: Uses `lib.logger` for TensorBoard integration
- **Loss Functions**: Uses `lib.loss_functions` for CrossEntropy loss

### Extension Points
- **Custom Models**: Extend Experiment class for new CNN architectures
- **Custom Datasets**: Modify dataset.py for different image datasets
- **Custom Metrics**: Add classification-specific visualizations

## Future Enhancements

### Planned Features
- **Visualization**: Confusion matrix and per-class accuracy plots
- **Data Augmentation**: Random crops, flips, and color transformations
- **Advanced Architectures**: ResNet, VGG, and modern CNN variants
- **Transfer Learning**: Pre-trained model fine-tuning

### Extension Opportunities
- **Other Datasets**: CIFAR-100, ImageNet, custom datasets
- **Object Detection**: Bounding box prediction tasks
- **Segmentation**: Pixel-level classification
- **Ensemble Methods**: Model averaging and voting

## Dependencies

```bash
pip install torch torchvision tensorboard numpy
```

## Advanced Usage

For advanced usage patterns including custom architectures, data augmentation, and ensemble methods, see the comprehensive examples in the code and configuration options.