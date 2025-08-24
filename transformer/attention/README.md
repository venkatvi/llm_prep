# Attention Mechanisms

Comprehensive implementation of attention mechanisms for transformer models, supporting different efficiency trade-offs for production and research use cases.

## Overview

This module provides three types of attention mechanisms with different parameter counts, memory usage, and computational efficiency characteristics:

- **Multi-Head Attention (MHA)**: Standard transformer attention with maximum representational capacity
- **Multi-Query Attention (MQA)**: Memory-efficient attention with shared key/value heads  
- **Group Query Attention (GQA)**: Balanced approach grouping query heads to share key/value heads

## Files

- **`mha.py`** - Multi-Head Attention implementation
- **`mqa.py`** - Multi-Query Attention implementation  
- **`gqa.py`** - Group Query Attention implementation
- **`sdpa.py`** - Scaled Dot-Product Attention core function

## Attention Types Comparison

| Feature | MHA | GQA | MQA |
|---------|-----|-----|-----|
| **Query Heads** | H | H | H |
| **Key Heads** | H | G | 1 |
| **Value Heads** | H | G | 1 |
| **Parameters** | Highest | Medium | Lowest |
| **Memory (Inference)** | Highest | Medium | Lowest |
| **Speed** | Slowest | Medium | Fastest |
| **Quality** | Best | Good | Good |

Where:
- H = number of attention heads
- G = number of groups (G < H for GQA)

## Usage

### Multi-Head Attention (MHA)
```python
from transformer.attention.mha import MultiHeadAttention

# Standard transformer attention
mha = MultiHeadAttention(
    embed_dim=512,
    num_heads=8,
    apply_causal_mask=False  # True for autoregressive tasks
)

# Input: [batch_size, seq_len, embed_dim]
x = torch.randn(32, 128, 512)
output = mha(x)  # [32, 128, 512]
```

**Best for**: Small models, research, maximum quality requirements

### Multi-Query Attention (MQA)
```python
from transformer.attention.mqa import MultiQueryAttention

# Memory-efficient attention with single K/V heads
mqa = MultiQueryAttention(
    embed_dim=512,
    num_heads=8,
    apply_causal_mask=False
)

x = torch.randn(32, 128, 512)
output = mqa(x)  # [32, 128, 512]
```

**Best for**: Large models, inference-heavy applications, memory constraints

### Group Query Attention (GQA)
```python
from transformer.attention.gqa import GroupQueryAttention

# Balanced approach with grouped heads
gqa = GroupQueryAttention(
    embed_dim=512,
    num_heads=8,
    num_groups=4,  # 2 heads per group
    apply_causal_mask=False
)

x = torch.randn(32, 128, 512)
output = gqa(x)  # [32, 128, 512]
```

**Best for**: Production systems requiring balance between quality and efficiency

### Scaled Dot-Product Attention
```python
from transformer.attention.sdpa import scaled_dot_product_attention

# Core attention computation used by all mechanisms
query = torch.randn(32, 8, 128, 64)  # [batch, heads, seq_len, head_dim]
key = torch.randn(32, 8, 128, 64)
value = torch.randn(32, 8, 128, 64)

# Optional causal mask for autoregressive models
causal_mask = torch.triu(torch.ones(128, 128), diagonal=1).bool()

attention_output = scaled_dot_product_attention(
    query, key, value, 
    mask=causal_mask,
    dropout_p=0.1
)
```

## Mathematical Foundation

### Scaled Dot-Product Attention
The core attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q: Query matrix [batch, heads, seq_len, head_dim]  
- K: Key matrix [batch, heads, seq_len, head_dim]
- V: Value matrix [batch, heads, seq_len, head_dim]
- d_k: Dimension of key vectors (head_dim)

### Multi-Head Attention
MHA creates H parallel attention heads:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_H)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Multi-Query Attention  
MQA shares single K and V across all query heads:
```
MultiQuery(Q, K, V) = Concat(head_1, ..., head_H)W^O

where head_i = Attention(QW_i^Q, KW^K, VW^V)
```

### Group Query Attention
GQA groups query heads to share K/V:
```
GroupQuery(Q, K, V) = Concat(head_1, ..., head_H)W^O

where head_i = Attention(QW_i^Q, KW_{g(i)}^K, VW_{g(i)}^V)
and g(i) maps head i to its group
```

## Implementation Details

### Common Features
All attention mechanisms support:
- **Causal Masking**: For autoregressive/decoder models
- **Dropout**: Attention weight dropout for regularization  
- **Scaling**: Proper attention weight scaling by √d_k
- **Residual Connections**: When used in transformer layers
- **Batch Processing**: Efficient parallel computation

### Parameter Initialization
- Query, Key, Value projections: Xavier/Glorot uniform initialization
- Output projection: Xavier uniform initialization  
- Bias terms: Zero initialization

### Numerical Stability
- Attention weights scaled by √d_k to prevent softmax saturation
- Optional attention dropout to prevent overfitting
- Proper gradient flow through all linear projections

## Performance Characteristics

### Memory Usage (Inference)
For sequence length S, embed dimension D, H heads:

| Mechanism | Key/Value Cache | Total Parameters |
|-----------|-----------------|------------------|
| **MHA** | 2 × H × S × (D/H) | 4 × D² |
| **GQA** | 2 × G × S × (D/H) | D² × (2 + 2G/H) |  
| **MQA** | 2 × 1 × S × (D/H) | 3 × D² |

### Computational Complexity
All mechanisms have O(S²D) attention computation complexity, but differ in:
- Linear projection costs
- Key/Value cache memory access patterns
- Parallel computation efficiency

## Integration

These attention mechanisms integrate seamlessly with:
- **Transformer Encoder**: Via `transformer/encoder.py`
- **Transformer Decoder**: Via `transformer/decoder.py` with cross-attention support
- **Complete Models**: Via `transformer/transformer_model.py` architecture selection
- **Regression Tasks**: Via `regression/h_transformer.py` wrapper models

## Configuration

Attention mechanisms can be selected via configuration:

```python
from transformer.attention_utils import get_attention

# Unified interface for all attention types
attention = get_attention(
    attention_type="gqa",  # "mha", "mqa", or "gqa"
    embed_dim=512,
    num_heads=8,
    num_groups=4,  # Only used for GQA
    apply_causal_mask=False
)
```

## Testing

Comprehensive test coverage in `transformer/tests/test_attention.py`:
- Forward pass shape validation
- Gradient computation correctness
- Attention weight properties
- Causal masking behavior
- Parameter count differences
- Cross-attention compatibility

Run tests:
```bash
cd transformer/tests
pytest test_attention.py -v
```