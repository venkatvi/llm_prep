# PyTorch Machine Learning Framework

Comprehensive ML framework with regression, classification, transformer models, and custom autograd implementations. Features complete CI/CD infrastructure, extensive testing, and production-ready components.

## 🏗️ Structure

- **`regression/`** - Linear, non-linear, and transformer regression with experiment management
- **`classification/`** - CIFAR-10 CNN classification with data pipelines
- **`autograd/`** - Custom PyTorch autograd implementations (educational)
- **`transformer/`** - Complete transformer architecture with encoder, decoder, and encoder-decoder models supporting causal masking and sequence-to-sequence tasks
- **`lib/`** - Core library components (configs, training, logging, utils)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run comprehensive tests
python run_all_tests.py

# Linear regression
cd regression && python main.py --type linear --epochs 1000

# Non-linear MLP regression  
cd regression && python main.py --type nlinear --epochs 1000 --latent_dims "128,64,32"

# Transformer regression (MHA attention)
cd regression && python main.py --type transformer --epochs 1000

# Transformer with Multi-Query Attention (MQA)
cd regression && python main.py --type transformer --attention_type mqa --epochs 1000

# Transformer with Group Query Attention (GQA)  
cd regression && python main.py --type transformer --attention_type gqa --epochs 1000

# Transformer autoregressive
cd regression && python main.py --type transformer --autoregressive --epochs 1000

# Transformer encoder-decoder
cd regression && python main.py --type transformer --encoderdecoder --epochs 1000

# CIFAR-10 classification
cd classification && python main.py

# Custom autograd experiments
cd autograd && python main.py
```

## ✨ Features

- **🤖 Models**: Linear regression, MLP, Transformer (regression + autoregressive + encoder-decoder), CNN for CIFAR-10
- **🎯 Attention Mechanisms**: Multiple attention types (MHA, MQA, GQA) for efficiency and performance trade-offs
- **🔧 Training**: Complete pipelines with validation, optimizers, schedulers
- **⚙️ Experiment Management**: Structured configs, hyperparameter sweeps
- **📊 Logging**: TensorBoard integration with visualization
- **🎓 Custom Autograd**: Educational gradient computation implementations
- **🧪 Testing**: Comprehensive test suite with 72+ tests
- **🔄 CI/CD**: GitHub Actions workflows with automated testing
- **📈 Data Generation**: Synthetic polynomial and sequence data utilities
- **🔧 Type Safety**: Comprehensive type annotations following PEP 484/585 standards
- **🏗️ Architecture**: Cross-attention support for sequence-to-sequence modeling

## 📋 Usage Examples

### Regression Models
```python
from lib.configs import ExperimentConfig, TrainConfig
from regression.experiment import RegressionExperiment

# Non-linear MLP
config = ExperimentConfig(
    type='nlinear',
    train_config=TrainConfig(epochs=1000, optimizer='adam', lr=0.001)
)
experiment = RegressionExperiment(config)
experiment.train()
predictions = experiment.predict()
```

### Transformer Models
```python
from regression.configs import TransformerModelConfig, AutoregressiveDecodeConfig
from regression.h_transformer import RegressionTransformerModel, ARTransformerModel
from regression.experiment import TransformerExperiment

# Regression mode (scalar prediction)
config = TransformerModelConfig(
    name="transformer_reg",
    input_dim=8, embed_dim=32, ffn_latent_dim=128,
    num_layers=2, num_heads=4, output_dim=1,
    apply_causal_mask=False, autoregressive_mode=False,
    attention_type="mha"  # "mha", "mqa", or "gqa"
)
model = RegressionTransformerModel(config)
inputs, targets = model.generate_data(random_seed=42)
predictions = model(inputs)

# Autoregressive mode (sequence generation)
ar_config = TransformerModelConfig(
    name="transformer_ar",
    input_dim=1, embed_dim=64, ffn_latent_dim=128,
    num_layers=2, num_heads=2, output_dim=1,
    apply_causal_mask=True, autoregressive_mode=True,
    decode_config=AutoregressiveDecodeConfig(
        num_steps=10, expanding_context=True, max_seq_len=40
    )
)
experiment = TransformerExperiment(experiment_config, autoregressive=True)
experiment.train()
generated_tokens = experiment.predict_autoregressively(input_sequence)
```

### Encoder-Decoder Transformers
```python
from regression.configs import EncoderDecoderConfig, AutoregressiveDecodeConfig
from regression.h_transformer import EncoderDecoderWrapper
from regression.experiment import EncoderDecoderExperiment

# Sequence-to-sequence configuration
config = EncoderDecoderConfig(
    name="encoder_decoder",
    input_dim=1, embed_dim=64, ffn_latent_dim=128,
    num_encoder_layers=2, num_decoder_layers=2, num_heads=2, output_dim=1,
    apply_causal_mask=True, autoregressive_mode=True,
    decode_config=AutoregressiveDecodeConfig(
        num_steps=10, expanding_context=True, max_seq_len=40
    )
)

# Train encoder-decoder model
experiment = EncoderDecoderExperiment(experiment_config)
experiment.train()

# Generate target sequences from source sequences
generated_sequence = experiment.predict_encoder_decoder()
```

### CIFAR-10 Classification
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

### Custom Autograd Operations
```python
import torch
from autograd.linear import Linear
from autograd.activations import ReLU
from autograd.simple import Square, Power

x = torch.randn(2, 3, requires_grad=True)
w = torch.randn(4, 3, requires_grad=True)
b = torch.randn(2, 4, requires_grad=True)
n = torch.tensor(2.0, requires_grad=True)

# Custom operations with gradient support
y = Linear.apply(x, w, b)  # Custom linear layer
z = ReLU.apply(y)          # Custom ReLU
p = Power.apply(z, n)      # Custom power function
loss = p.sum()
loss.backward()
```

### Synthetic Data Generation
```python
from regression.dataset import generate_polynomial_data

# Generate polynomial regression data
inputs, targets = generate_polynomial_data(
    num_samples=1000,
    degree=3,           # Cubic polynomial
    noise_level=0.1,
    x_range=(0.0, 5.0),
    random_seed=42
)
```

### Attention Mechanisms

The framework supports three types of attention mechanisms with different efficiency trade-offs:

```python
from transformer.attention_utils import get_attention

# Multi-Head Attention (MHA) - Standard transformer attention
mha = get_attention("mha", embed_dim=64, num_heads=8, num_groups=4, apply_causal_mask=False)

# Multi-Query Attention (MQA) - Single K/V heads, multiple Q heads  
mqa = get_attention("mqa", embed_dim=64, num_heads=8, num_groups=4, apply_causal_mask=False)

# Group Query Attention (GQA) - Grouped K/V heads for balance
gqa = get_attention("gqa", embed_dim=64, num_heads=8, num_groups=4, apply_causal_mask=False)
```

**Attention Types Comparison:**

| Type | Parameters | Memory Usage | Inference Speed | Use Case |
|------|------------|--------------|-----------------|----------|
| **MHA** | Highest | Highest | Slowest | Maximum quality, small models |
| **GQA** | Medium | Medium | Medium | Balanced performance/efficiency |
| **MQA** | Lowest | Lowest | Fastest | Large models, inference-heavy |

**CLI Usage:**
```bash
# Use different attention types via command line
python main.py --type transformer --attention_type mha  # Multi-Head Attention
python main.py --type transformer --attention_type mqa  # Multi-Query Attention  
python main.py --type transformer --attention_type gqa  # Group Query Attention
```

## 🧪 Testing

Comprehensive test suite with 72+ tests covering all components:

```bash
# Run all tests
python run_all_tests.py

# Run specific module tests
cd autograd/tests && python run_tests.py
cd transformer/tests && python run_tests.py

# Run with pytest
pytest autograd/tests/ -v
pytest transformer/tests/ -v
```

**Test Coverage:**
- ✅ Module imports and compatibility
- ✅ Custom autograd implementations (72 tests)
- ✅ Transformer components (50+ tests)
- ✅ Cross-module integration
- ✅ Training pipelines and data generation

## 🔧 Development

### CI/CD Pipeline
- **GitHub Actions**: Automated testing on Python 3.9-3.12
- **Pre-commit Hooks**: Code formatting and quality checks
- **Make Commands**: Development workflow automation

```bash
# Development commands
make install    # Install dependencies
make test      # Run all tests
make lint      # Code formatting
make clean     # Clean build artifacts
```

### Project Commands
```bash
# Hyperparameter sweeps
cd regression && python experiment_sweep.py

# TensorBoard visualization
tensorboard --logdir regression/logs

# Model checkpoints and predictions
ls regression/logs/  # View experiment results
```

## 📚 Documentation

### Core Modules
- **[`regression/README.md`](regression/README.md)** - Regression models and experiments
- **[`classification/README.md`](classification/README.md)** - CIFAR-10 classification
- **[`autograd/README.md`](autograd/README.md)** - Custom autograd implementations
- **[`transformer/README.md`](transformer/README.md)** - Transformer architectures
- **[`transformer/attention/README.md`](transformer/attention/README.md)** - Attention mechanisms (MHA, MQA, GQA)
- **[`lib/README.md`](lib/README.md)** - Core library utilities

### Testing & CI/CD
- **[`autograd/tests/README.md`](autograd/tests/README.md)** - Custom autograd test suite
- **[`transformer/tests/README.md`](transformer/tests/README.md)** - Transformer component tests  
- **[`regression/tests/README.md`](regression/tests/README.md)** - Regression model test suite
- **[`TESTING.md`](TESTING.md)** - Testing infrastructure and guidelines
- **[`CI_SETUP_SUMMARY.md`](CI_SETUP_SUMMARY.md)** - CI/CD setup and workflows

## 🎯 Key Highlights

- **🏭 Production Ready**: Complete CI/CD, testing, and comprehensive documentation
- **🔬 Educational**: Custom autograd for understanding PyTorch internals  
- **🚀 Modern Architecture**: Full transformer stack (encoder, decoder, encoder-decoder) with cross-attention and causal masking
- **🔄 Sequence-to-Sequence**: Encoder-decoder models for translation and sequence generation tasks
- **📊 Experiment Tracking**: TensorBoard integration with structured configs
- **🔧 Type Safety**: Comprehensive type annotations with mypy compatibility
- **🎯 Reproducible**: Fixed seeds, deterministic training, state management
- **⚡ Efficient**: DataLoader support, batch processing, GPU compatibility