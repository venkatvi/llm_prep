# Transformer Components

PyTorch transformer encoder implementation for sequence processing.

## Features

- **Multi-Head Attention**: Scaled dot-product attention mechanism
- **Feedforward Network**: Two-layer MLP with ReLU activation
- **Positional Encoding**: Sinusoidal position embeddings
- **Layer Normalization**: Post-norm transformer architecture

## Components

- **`transformer_model.py`** - Complete transformer encoder
- **`attention.py`** - Multi-head self-attention
- **`ffn.py`** - Feedforward network
- **`input_encodings.py`** - Positional encoding

## Architecture

```
Input: [batch, seq_len, input_dim]
├── Linear Projection → [batch, seq_len, embed_dim]
├── Positional Encoding
├── Transformer Layers (N layers)
│   ├── Multi-Head Attention + Residual
│   ├── Layer Norm
│   ├── Feedforward Network + Residual
│   └── Layer Norm
├── Global Average Pooling → [batch, embed_dim]
└── Output Projection → [batch, output_dim]
```

## Usage

```python
from transformer.transformer_model import TransformerModel

model = TransformerModel(
    input_dim=8,
    embed_dim=32,
    ffn_latent_dim=128,
    num_layers=2,
    num_heads=2,
    output_dim=1
)

# Input: [batch_size, sequence_length, input_dim]
x = torch.randn(32, 16, 8)
output = model(x)  # [32, 1]
```

## Configuration

| Parameter | Description |
|-----------|-------------|
| `input_dim` | Input feature dimension |
| `embed_dim` | Embedding/hidden dimension |
| `ffn_latent_dim` | Feedforward network hidden size |
| `num_layers` | Number of transformer layers |
| `num_heads` | Number of attention heads |
| `output_dim` | Output dimension |

## Integration

Used in regression tasks via `regression/h_transformer.py` wrapper for sequence-to-scalar prediction.