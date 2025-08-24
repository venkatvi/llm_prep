# Transformer Components

PyTorch transformer implementation with complete encoder, decoder, and encoder-decoder architectures supporting regression, autoregressive, and sequence-to-sequence tasks.

## Features

- **Multi-Head Attention**: Scaled dot-product attention with optional causal masking
- **Multi-Query Attention (MQA)**: Memory-efficient attention with single key/value heads
- **Group Query Attention (GQA)**: Balanced attention mechanism grouping heads for efficiency
- **Cross-Attention**: Encoder-decoder attention for sequence-to-sequence modeling
- **Causal Masking**: Support for autoregressive generation with future token masking
- **Feedforward Network**: Two-layer MLP with ReLU activation
- **Positional Encoding**: Sinusoidal position embeddings
- **Layer Normalization**: Post-norm transformer architecture
- **Multiple Architectures**: Regression (pooled), autoregressive (sequence), and encoder-decoder modes
- **Attention Factory**: Unified interface for different attention mechanisms
- **Type Annotations**: Comprehensive type hints for all components

## Components

- **`transformer_model.py`** - Complete transformer models (regression + autoregressive + encoder-decoder)
- **`decoder.py`** - Transformer decoder layer with cross-attention support
- **`encoder.py`** - Transformer encoder layer
- **`attention/`** - Attention mechanism implementations:
  - **`mha.py`** - Multi-Head Attention (standard transformer attention)
  - **`mqa.py`** - Multi-Query Attention (single K/V heads for efficiency)
  - **`gqa.py`** - Group Query Attention (grouped K/V heads for balance)
- **`attention_utils.py`** - Attention factory and utilities
- **`ffn.py`** - Feedforward network
- **`input_encodings.py`** - Positional encoding

## Architecture

### Regression Mode (TransformerModel)
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

### Autoregressive Mode (AutoregressiveTransformerModel)
```
Input: [batch, seq_len, input_dim]
├── Linear Projection → [batch, seq_len, embed_dim]
├── Positional Encoding
├── Transformer Layers (N layers) with Causal Masking
│   ├── Causal Multi-Head Attention + Residual
│   ├── Layer Norm
│   ├── Feedforward Network + Residual
│   └── Layer Norm
└── Output Projection → [batch, seq_len, output_dim]
```

### Encoder-Decoder Mode (EncoderDecoder)
```
Source: [batch, src_len, input_dim]    Target: [batch, tgt_len, input_dim]
│                                      │
├── Encoder                           ├── Decoder
│   ├── Linear Projection             │   ├── Linear Projection
│   ├── Positional Encoding          │   ├── Positional Encoding
│   └── Encoder Layers (N layers)    │   └── Decoder Layers (M layers)
│       ├── Multi-Head Attention     │       ├── Causal Self-Attention
│       ├── Layer Norm               │       ├── Layer Norm
│       ├── FFN + Residual           │       ├── Cross-Attention (to encoder)
│       └── Layer Norm               │       ├── Layer Norm
│                                     │       ├── FFN + Residual
└── Encoder Output ──────────────────┘       └── Layer Norm
    [batch, src_len, embed_dim]               │
                                             └── Output Projection
                                                 [batch, tgt_len, output_dim]
```

## Usage

### Regression Model
```python
from transformer.transformer_model import TransformerModel

model = TransformerModel(
    input_dim=8,
    embed_dim=32,
    ffn_latent_dim=128,
    num_layers=2,
    num_heads=2,
    output_dim=1,
    apply_causal_mask=False,
    max_seq_len=128,
    attention_type="mha"  # "mha", "mqa", or "gqa"
)

# Input: [batch_size, sequence_length, input_dim]
x = torch.randn(32, 16, 8)
output = model(x)  # [32, 1] - scalar prediction
```

### Autoregressive Model
```python
from transformer.transformer_model import AutoregressiveTransformerModel

model = AutoregressiveTransformerModel(
    input_dim=1,
    embed_dim=64,
    ffn_latent_dim=128,
    num_layers=2,
    num_heads=2,
    output_dim=1,
    apply_causal_mask=True,
    max_seq_len=128
)

# Input: [batch_size, sequence_length, input_dim]
x = torch.randn(32, 16, 1)
output = model(x)  # [32, 16, 1] - full sequence

# Next token generation
next_token = model.generate_next_token(x)  # [32, 1, 1]
```

### Encoder-Decoder Model
```python
from transformer.transformer_model import EncoderDecoder

model = EncoderDecoder(
    input_dim=1,
    embed_dim=64,
    ffn_latent_dim=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=2,
    output_dim=1,
    apply_causal_mask=True,
    max_seq_len=128
)

# Source and target sequences
source = torch.randn(32, 16, 1)      # [batch, src_len, input_dim]
target = torch.randn(32, 20, 1)      # [batch, tgt_len, input_dim]

# Encode source sequence
encoder_output = model.encode(source)  # [32, 16, 64]

# Decode with target (teacher forcing during training)
decoder_output = model.decode(target, encoder_output)  # [32, 20, 1]

# Full forward pass
output = model(source, target)  # [32, 20, 1]
```

### Causal Masking
```python
from transformer.attention import create_causal_mask, scaled_dot_product_attention

# Create causal mask for sequence length 10
mask = create_causal_mask(10)  # Lower triangular matrix

# Use in attention computation
attention_output = scaled_dot_product_attention(
    query, key, value, mask=mask
)
```

## Configuration

| Parameter | Description |
|-----------|-------------|
| `input_dim` | Input feature dimension |
| `embed_dim` | Embedding/hidden dimension |
| `ffn_latent_dim` | Feedforward network hidden size |
| `num_layers` | Number of transformer layers (encoder-only) |
| `num_encoder_layers` | Number of encoder layers (encoder-decoder) |
| `num_decoder_layers` | Number of decoder layers (encoder-decoder) |
| `num_heads` | Number of attention heads |
| `output_dim` | Output dimension |
| `apply_causal_mask` | Enable causal masking for autoregressive tasks |
| `max_seq_len` | Maximum sequence length for positional encoding |
| `attention_type` | Type of attention mechanism ("mha", "mqa", "gqa") |

## Attention Mechanisms

The transformer supports three types of attention mechanisms, each with different efficiency characteristics:

### Multi-Head Attention (MHA)
```python
from transformer.attention.mha import MultiHeadAttention

attn = MultiHeadAttention(embed_dim=64, num_heads=8, apply_causal_mask=False)
```

**Standard transformer attention with multiple independent query, key, and value heads.**
- ✅ Maximum representational capacity
- ❌ Highest memory usage during inference
- ❌ Slowest inference speed
- 🎯 Best for: Small models, maximum quality requirements

### Multi-Query Attention (MQA)
```python
from transformer.attention.mqa import MultiQueryAttention

attn = MultiQueryAttention(embed_dim=64, num_heads=8)
```

**Memory-efficient attention with multiple query heads but single key/value heads.**
- ✅ Lowest memory usage during inference
- ✅ Fastest inference speed
- ❌ Slight quality reduction compared to MHA
- 🎯 Best for: Large models, inference-heavy applications

### Group Query Attention (GQA)
```python
from transformer.attention.gqa import GroupQueryAttention

attn = GroupQueryAttention(embed_dim=64, num_heads=8, num_groups=4)
```

**Balanced approach that groups query heads to share key/value heads.**
- ⚖️ Balanced memory usage and quality
- ⚖️ Medium inference speed
- ✅ Good compromise between MHA and MQA
- 🎯 Best for: Production models requiring balance

### Attention Factory
```python
from transformer.attention_utils import get_attention

# Unified interface for all attention types
mha = get_attention("mha", embed_dim=64, num_heads=8, num_groups=4, apply_causal_mask=False)
mqa = get_attention("mqa", embed_dim=64, num_heads=8, num_groups=4, apply_causal_mask=False)
gqa = get_attention("gqa", embed_dim=64, num_heads=8, num_groups=4, apply_causal_mask=False)
```

### Performance Comparison

| Attention Type | Parameters | Memory (Inference) | Speed | Quality | Use Case |
|----------------|------------|-------------------|--------|---------|----------|
| **MHA** | Highest | Highest | Slowest | Best | Small models, research |
| **GQA** | Medium | Medium | Medium | Good | Production systems |
| **MQA** | Lowest | Lowest | Fastest | Good | Large models, inference |

## Integration

- **Regression Tasks**: Via `regression/h_transformer.py` wrapper for sequence-to-scalar prediction
- **Autoregressive Tasks**: Via `regression/h_transformer.py` for next-token prediction and generation
- **Sequence-to-Sequence Tasks**: Via `regression/h_transformer.py` EncoderDecoderWrapper for translation tasks
- **Experiment Framework**: Integrated with `regression/experiment.py` for all modes (regression, autoregressive, encoder-decoder)

## Key Differences

| Feature | Regression Model | Autoregressive Model | Encoder-Decoder Model |
|---------|------------------|----------------------|----------------------|
| **Output** | Scalar (pooled) | Full sequence | Target sequence |
| **Masking** | None | Causal masking | Causal in decoder |
| **Use Case** | Classification/Regression | Sequence generation | Seq2seq translation |
| **Pooling** | Global average | None | None |
| **Training** | Supervised learning | Next-token prediction | Teacher forcing |
| **Architecture** | Encoder only | Encoder only | Encoder + Decoder |
| **Cross-Attention** | No | No | Yes (decoder to encoder) |