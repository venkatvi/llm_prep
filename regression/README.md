# Regression Models

PyTorch regression framework with linear, non-linear, and transformer models supporting regression, autoregressive prediction, and sequence-to-sequence modeling with encoder-decoder architectures.

## Features

- **Models**: Linear regression, MLP, and Transformer (regression + autoregressive + encoder-decoder)
- **Training**: Complete pipeline with validation, optimizers, schedulers
- **Data**: Synthetic data generation with DataLoader support
- **Autoregressive**: Token-by-token generation with causal masking
- **Sequence-to-Sequence**: Encoder-decoder architecture with cross-attention for translation tasks
- **Logging**: TensorBoard integration with prediction visualization
- **Type Safety**: Comprehensive type annotations for better code quality

## Quick Start

```bash
# Linear regression
python main.py --type linear --epochs 1000 --lr 0.01

# Non-linear MLP
python main.py --type nlinear --epochs 1000 --lr 0.001 --latent_dims "128,64,32"

# Transformer regression
python main.py --type transformer --epochs 1000 --lr 0.001

# Transformer autoregressive
python main.py --type transformer --autoregressive --epochs 1000 --lr 0.001

# Transformer encoder-decoder (sequence-to-sequence)
python main.py --type transformer --encoderdecoder --epochs 1000 --lr 0.001

# Hyperparameter sweep
python experiment_sweep.py
```

## Files

- **`main.py`** - CLI entry point with support for `--encoderdecoder` flag
- **`experiment.py`** - RegressionExperiment, TransformerExperiment, and EncoderDecoderExperiment orchestrators
- **`e_linear_reg.py`** - Linear model implementation
- **`e_non_linear_reg.py`** - MLP model implementation
- **`h_transformer.py`** - Transformer wrappers (regression, autoregressive, encoder-decoder)
- **`configs.py`** - Model configurations including EncoderDecoderConfig
- **`dataset.py`** - Dataset utilities and data generation

## Usage

```python
from lib.configs import ExperimentConfig, TrainConfig
from experiment import RegressionExperiment

config = ExperimentConfig(
    type="nlinear",
    train_config=TrainConfig(epochs=1000, optimizer="adam", lr=0.001)
)

experiment = RegressionExperiment(config)
experiment.train()
predictions = experiment.predict()
```

## Key Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--type` | `linear`, `nlinear`, `transformer` | Model type |
| `--autoregressive` | flag | Enable transformer autoregressive mode |
| `--encoderdecoder` | flag | Use encoder-decoder architecture for seq2seq tasks |
| `--epochs` | integer | Training epochs |
| `--lr` | float | Learning rate |
| `--optimizer` | `adam`, `sgd`, `rmsprop` | Optimizer |
| `--custom_act` | `relu`, `tanh`, `gelu` | Activation |
| `--latent_dims` | comma-separated | Hidden dimensions |

## Transformer Modes

### Regression Mode (default)
- Uses global average pooling for scalar prediction
- Suitable for sequence-to-value tasks
- Example: predicting sum of sequence elements

### Autoregressive Mode (`--autoregressive`)
- Enables next-token prediction with causal masking
- Supports expanding context or sliding window generation
- Uses sequence-to-sequence architecture

### Encoder-Decoder Mode (`--encoderdecoder`)
- Full sequence-to-sequence architecture with separate encoder and decoder
- Cross-attention mechanism connecting encoder outputs to decoder
- Causal masking in decoder for autoregressive generation
- Suitable for translation, summarization, and other seq2seq tasks
- Independent positional encodings for encoder and decoder sequences