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
- **`configs.py`** - Model configurations including EncoderDecoderConfig, TransformerModelConfig, and FFNConfig
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
| `--attention_type` | `mha`, `mqa`, `gqa` | Attention mechanism type |
| `--epochs` | integer | Training epochs |
| `--lr` | float | Learning rate |
| `--optimizer` | `adam`, `sgd`, `rmsprop` | Optimizer |
| `--custom_act` | `relu`, `tanh`, `gelu` | Activation |
| `--latent_dims` | comma-separated | Hidden dimensions |
| `--moe` | flag | Enable Mixture of Experts in FFN layers |

## Feed-Forward Network Configuration

The FFN layers in transformer models can be configured with standard FFN or Mixture of Experts (MOE):

### Standard FFN
```python
from regression.configs import FFNConfig

ffn_config = FFNConfig(
    embed_dim=128,
    latent_dim=512,
    use_moe=False  # Standard FFN
)
```

### Mixture of Experts (MOE)
```python
ffn_config = FFNConfig(
    embed_dim=128,
    latent_dim=512,
    use_moe=True,
    num_experts=8,      # Number of expert networks
    capacity=64,        # Token capacity per expert
    alpha=0.01,         # Load balancing loss weight
    topk=2              # Number of experts to activate per token
)
```

**MOE Benefits:**
- **Scalability**: Add parameters without increasing compute
- **Specialization**: Experts learn different patterns
- **Efficiency**: Only activates subset of parameters per token

**MOE Usage**: Enable with `--moe` flag or set `use_moe=True` in FFNConfig

## Attention Mechanisms

The transformer models support three types of attention mechanisms:

### Multi-Head Attention (MHA)
- **Usage**: `--attention_type mha` (default)
- **Configuration**: `num_heads=4`, `num_groups=2` (groups parameter ignored)
- **Best for**: Maximum quality, small models

### Multi-Query Attention (MQA) 
- **Usage**: `--attention_type mqa`
- **Configuration**: `num_heads=4`, `num_groups=1` (groups parameter ignored)
- **Best for**: Large models, inference optimization

### Group Query Attention (GQA)
- **Usage**: `--attention_type gqa` 
- **Configuration**: `num_heads=4`, `num_groups=2` (configurable)
- **Best for**: Balance between quality and efficiency

### num_groups Configuration

The `num_groups` parameter is configurable in `TransformerModelConfig` and `EncoderDecoderConfig`:

```python
from regression.configs import TransformerModelConfig, AutoregressiveDecodeConfig, FFNConfig

# Configure Feed-Forward Network
ffn_config = FFNConfig(
    embed_dim=64,
    latent_dim=128,
    use_moe=False  # Enable for Mixture of Experts
)

# Example configurations for different attention types
config = TransformerModelConfig(
    name="transformer",
    max_seq_len=128, input_dim=1, embed_dim=64, ffn_latent_dim=128,
    num_layers=2, num_heads=8, num_groups=4,  # Configurable groups
    output_dim=1, apply_causal_mask=True, autoregressive_mode=True,
    decode_config=AutoregressiveDecodeConfig(use_kv_cache=True),
    attention_type="gqa",
    ffn_config=ffn_config
)

# num_groups configurations:
# num_groups=1: GQA acts like MQA (single shared K/V)
# num_groups=4: 2 query heads per group (balanced)
# num_groups=8: GQA acts like MHA (no sharing)
```

### Performance Trade-offs

| Configuration | Memory Usage | Inference Speed | Quality | Use Case |
|---------------|--------------|-----------------|---------|----------|
| `mha` + `num_groups=8` | Highest | Slowest | Best | Research, small models |
| `gqa` + `num_groups=4` | Medium | Medium | Good | Production systems |
| `gqa` + `num_groups=2` | Lower | Faster | Good | Efficiency-focused |
| `gqa` + `num_groups=1` | Lowest | Fastest | Good | MQA-equivalent |
| `mqa` + `num_groups=1` | Lowest | Fastest | Good | Large models |

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