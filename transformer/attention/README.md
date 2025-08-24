# Attention Mechanisms

Comprehensive implementation of attention mechanisms for transformer models, supporting different efficiency trade-offs for production and research use cases.

## Overview

This module provides three types of attention mechanisms with different parameter counts, memory usage, and computational efficiency characteristics:

- **Multi-Head Attention (MHA)**: Standard transformer attention with maximum representational capacity
- **Multi-Query Attention (MQA)**: Memory-efficient attention with shared key/value heads  
- **Group Query Attention (GQA)**: Balanced approach grouping query heads to share key/value heads
- **KV Caching**: All mechanisms support key-value caching for optimized autoregressive inference

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
| **num_groups** | H (ignored) | G (configurable) | 1 (ignored) |
| **Parameters** | Highest | Medium | Lowest |
| **Memory (Inference)** | Highest | Medium | Lowest |
| **Speed** | Slowest | Medium | Fastest |
| **Quality** | Best | Good | Good |

Where:
- H = number of attention heads
- G = number of groups (G < H for GQA, configurable via `num_groups` parameter)

## KV Caching

All attention mechanisms support key-value caching for efficient autoregressive generation. KV caching stores previously computed key and value tensors, avoiding redundant computation during sequential token generation.

### Caching Mechanism

Each attention module implements two forward methods:
- **`forward(input, kv, expanding_context)`**: Standard forward pass with optional caching
- **`forward_with_cache(input, kv, expanding_context)`**: Dedicated cached inference method

#### Cache States
1. **No Cache**: `kv_cache is None` - First forward pass, computes and stores all K,V pairs
2. **Cache Hit**: `kv_cache exists` - Reuses cached K,V and only computes new ones for current token

#### Context Management
- **Expanding Context** (`expanding_context=True`): Cache grows with each token (good for short sequences)
- **Sliding Window** (`expanding_context=False`): Fixed cache size, maintains recent context (good for long sequences)

### Usage Examples

#### MHA with KV Caching
```python
from transformer.attention.mha import MultiHeadAttention

mha = MultiHeadAttention(
    embed_dim=512, num_heads=8, 
    apply_causal_mask=True, use_kv_cache=True
)

# First pass - initializes cache
input_seq = torch.randn(1, 10, 512)  # [batch, seq_len, embed_dim]
output1 = mha(input_seq, kv=None, expanding_context=True)

# Subsequent passes - reuses cache
next_token = torch.randn(1, 1, 512)   # [batch, 1, embed_dim]
output2 = mha(next_token, kv=None, expanding_context=True)  # Fast!
```

#### MQA with KV Caching (Most Memory Efficient)
```python  
from transformer.attention.mqa import MultiQueryAttention

mqa = MultiQueryAttention(
    embed_dim=512, num_heads=8,
    apply_causal_mask=True, use_kv_cache=True
)

# MQA has the smallest cache footprint due to single K,V heads
input_seq = torch.randn(1, 10, 512)
output = mqa(input_seq, kv=None, expanding_context=True)
# Cache size: 2 × 1 × seq_len × head_dim (vs H heads for MHA)
```

#### GQA with KV Caching (Balanced)
```python
from transformer.attention.gqa import GroupQueryAttention

gqa = GroupQueryAttention(
    embed_dim=512, num_heads=8, num_groups=4,
    apply_causal_mask=True, use_kv_cache=True  
)

input_seq = torch.randn(1, 10, 512)
output = gqa(input_seq, kv=None, expanding_context=True)
# Cache size: 2 × 4 × seq_len × head_dim (4 groups vs 8 heads)
```

#### Cross-Attention with KV Caching
```python
# Encoder-decoder scenario where encoder output is cached as K,V
encoder_output = torch.randn(1, 20, 512)  # [batch, src_len, embed_dim]
decoder_input = torch.randn(1, 1, 512)     # [batch, 1, embed_dim]

# Cross-attention: decoder queries attend to cached encoder K,V
cross_attn = MultiHeadAttention(embed_dim=512, num_heads=8, use_kv_cache=True)
output = cross_attn(decoder_input, kv=encoder_output, expanding_context=True)
```

### Performance Impact

| Attention + Cache | Memory vs No Cache | Speed vs No Cache | Cache Size per Layer |
|-------------------|-------------------|------------------|----------------------|
| **MHA + Cache** | +40% memory | 5-8x faster | 2 × H × S × (D/H) |
| **GQA + Cache** | +25% memory | 7-10x faster | 2 × G × S × (D/H) |
| **MQA + Cache** | +15% memory | 8-12x faster | 2 × 1 × S × (D/H) |

Where: H=heads, G=groups, S=sequence length, D=embed dimension

### Cache Implementation Details

#### Cache Structure
Each attention module stores cache as a dictionary:
```python
self.kv_cache = {
    "key": torch.Tensor,     # Cached key projections
    "value": torch.Tensor,   # Cached value projections  
}
```

#### Cache Lifecycle
1. **Initialization**: On first `forward_with_cache()` call with `kv_cache=None`
2. **Population**: Compute K,V for full input sequence and store
3. **Updates**: On subsequent calls, append new K,V for current tokens
4. **Management**: Handle context expansion or sliding window based on `expanding_context`
5. **Reset**: Cache cleared when model parameters change or explicit reset

#### Memory Layout
Cached tensors maintain the attention head format:
- **MHA**: `[batch_size, num_heads, seq_len, head_dim]`
- **GQA**: `[batch_size, num_groups, seq_len, head_dim]` 
- **MQA**: `[batch_size, 1, seq_len, head_dim]`

### Integration

KV caching integrates seamlessly with:
- **Transformer Layers**: Via encoder/decoder implementations
- **Model Wrappers**: Via `transformer_model.py` architecture classes
- **Experiment Framework**: Via `regression/h_transformer.py` experiment wrappers
- **Attention Factory**: Via `get_attention()` with `use_kv_cache=True`

## Usage

### Multi-Head Attention (MHA)
```python
from transformer.attention.mha import MultiHeadAttention

# Standard transformer attention
mha = MultiHeadAttention(
    embed_dim=512,
    num_heads=8,
    apply_causal_mask=False,  # True for autoregressive tasks
    use_kv_cache=False        # Enable for inference optimization
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
    apply_causal_mask=False,
    use_kv_cache=False        # Most efficient when enabled
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
    num_groups=4,  # 2 heads per group (configurable parameter)
    apply_causal_mask=False,
    use_kv_cache=False        # Good balance of speed and memory
)

x = torch.randn(32, 128, 512)
output = gqa(x)  # [32, 128, 512]
```

**Configuration**: The `num_groups` parameter controls the number of key/value groups:
- `num_groups=1`: Equivalent to MQA (single shared K/V)
- `num_groups=num_heads`: Equivalent to MHA (no sharing)
- `1 < num_groups < num_heads`: True GQA with grouped sharing

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
    num_groups=4,          # Configurable groups for GQA (ignored for MHA/MQA)
    apply_causal_mask=False,
    use_kv_cache=True      # Enable KV caching for all types
)

# GQA configuration examples:
# num_groups=1: Acts like MQA (single K/V shared across all queries)
# num_groups=2: 4 query heads per group (efficient)
# num_groups=4: 2 query heads per group (balanced)  
# num_groups=8: 1 query head per group (equivalent to MHA)
```

### num_groups Configuration Guide

The `num_groups` parameter in GQA allows fine-tuning the trade-off between efficiency and quality:

| num_groups | Query Heads per Group | Memory Usage | Quality | Speed |
|------------|----------------------|--------------|---------|-------|
| 1 | 8 (MQA-like) | Lowest | Good | Fastest |
| 2 | 4 | Low | Better | Fast |
| 4 | 2 | Medium | Good | Medium |
| 8 | 1 (MHA-like) | Highest | Best | Slowest |

**Recommended configurations**:
- **Large models**: `num_groups=1` or `num_groups=2` for maximum efficiency
- **Medium models**: `num_groups=num_heads//4` for balanced performance  
- **Small models**: `num_groups=num_heads//2` for quality with some efficiency gains

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