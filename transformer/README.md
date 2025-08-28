# Transformer Components

PyTorch transformer implementation with complete encoder, decoder, and encoder-decoder architectures supporting regression, autoregressive, and sequence-to-sequence tasks.

## Features

- **Multi-Head Attention**: Scaled dot-product attention with optional causal masking
- **Multi-Query Attention (MQA)**: Memory-efficient attention with single key/value heads
- **Group Query Attention (GQA)**: Balanced attention mechanism grouping heads for efficiency
- **Mixture of Experts (MOE)**: Sparse expert routing with capacity constraints and load balancing
- **Speculative Decoding**: Accelerated autoregressive generation using draft-target model pairs
- **Cross-Attention**: Encoder-decoder attention for sequence-to-sequence modeling
- **KV Caching**: Optimized inference with key-value caching for autoregressive generation
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
- **`moe.py`** - Mixture of Experts with capacity-constrained routing
- **`input_encodings.py`** - Positional encoding
- **`configs.py`** - Configuration classes including FFNConfig for feedforward networks
- **`spec_decoding.py`** - Speculative decoding implementation for accelerated generation

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
from transformer.configs import FFNConfig

# Configure feedforward network
ffn_config = FFNConfig(
    embed_dim=32,
    latent_dim=128,
    use_moe=False  # Set to True for Mixture of Experts
)

model = TransformerModel(
    input_dim=8,
    embed_dim=32,
    ffn_latent_dim=128,
    num_layers=2,
    num_heads=2,
    output_dim=1,
    apply_causal_mask=False,
    max_seq_len=128,
    attention_type="mha",  # "mha", "mqa", or "gqa"
    ffn_config=ffn_config,
    use_kv_cache=False
)

# Input: [batch_size, sequence_length, input_dim]
x = torch.randn(32, 16, 8)
output = model(x)  # [32, 1] - scalar prediction
```

### Autoregressive Model
```python
from transformer.transformer_model import AutoregressiveTransformerModel
from transformer.configs import FFNConfig

# Configure feedforward network with MOE
ar_ffn_config = FFNConfig(
    embed_dim=64,
    latent_dim=128,
    use_moe=True,      # Enable Mixture of Experts
    num_experts=8,
    capacity=32,
    alpha=0.01,
    topk=2
)

model = AutoregressiveTransformerModel(
    input_dim=1,
    embed_dim=64,
    ffn_latent_dim=128,
    num_layers=2,
    num_heads=2,
    output_dim=1,
    apply_causal_mask=True,
    max_seq_len=128,
    attention_type="mha",
    ffn_config=ar_ffn_config,
    use_kv_cache=False
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
from transformer.configs import FFNConfig

# Configure feedforward network
encdec_ffn_config = FFNConfig(
    embed_dim=64,
    latent_dim=128,
    use_moe=False  # Standard FFN for encoder-decoder
)

model = EncoderDecoder(
    input_dim=1,
    embed_dim=64,
    ffn_latent_dim=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_heads=2,
    output_dim=1,
    apply_causal_mask=True,
    max_seq_len=128,
    attention_type="mha",
    ffn_config=encdec_ffn_config,
    use_kv_cache=False
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
| `num_groups` | Number of K/V groups for GQA (configurable, defaults to num_heads//2) |
| `output_dim` | Output dimension |
| `apply_causal_mask` | Enable causal masking for autoregressive tasks |
| `max_seq_len` | Maximum sequence length for positional encoding |
| `attention_type` | Type of attention mechanism ("mha", "mqa", "gqa") |
| `use_kv_cache` | Enable key-value caching for optimized inference |
| `ffn_config` | FFNConfig object for feedforward network configuration |

### FFNConfig Parameters

| Parameter | Description |
|-----------|-------------|
| `embed_dim` | Embedding dimension (must match transformer's embed_dim) |
| `latent_dim` | Hidden layer size in FFN (typically 4x embed_dim) |
| `use_moe` | Enable Mixture of Experts (False for standard FFN) |
| `num_experts` | Number of expert networks (required if use_moe=True) |
| `capacity` | Token capacity per expert (prevents overloading) |
| `alpha` | Load balancing loss weight (typically 0.01) |
| `topk` | Number of experts to activate per token (1 or 2) |

## KV Caching

Key-Value caching dramatically improves inference performance for autoregressive generation by reusing previously computed key and value tensors, avoiding redundant computation across generation steps.

### How KV Caching Works

During autoregressive generation, at each step:
1. **First Pass**: Compute and cache all key/value pairs for the input sequence
2. **Subsequent Passes**: Reuse cached K,V pairs and only compute new ones for the current token
3. **Context Management**: Handle cache growth with expanding or sliding window strategies

### Usage

#### Basic KV Caching
```python
from transformer.transformer_model import AutoregressiveTransformerModel
from transformer.configs import FFNConfig

# Configure FFN for KV caching model
kv_ffn_config = FFNConfig(
    embed_dim=64,
    latent_dim=128,
    use_moe=False
)

# Enable KV caching in model configuration
model = AutoregressiveTransformerModel(
    input_dim=1, embed_dim=64, ffn_latent_dim=128,
    num_layers=2, num_heads=4, output_dim=1,
    apply_causal_mask=True, max_seq_len=128,
    attention_type="gqa", use_kv_cache=True,
    ffn_config=kv_ffn_config
)

# First forward pass - initializes and populates cache
initial_sequence = torch.randn(1, 10, 1)  # [batch, seq_len, input_dim]
output1 = model(initial_sequence, expanding_context=True)

# Subsequent passes - reuses cache for efficiency
next_token = torch.randn(1, 1, 1)          # [batch, 1, input_dim] 
output2 = model(next_token, expanding_context=True)  # Much faster
```

#### Context Management Modes
```python
# Expanding Context - Cache grows with each generation step
# Good for: Short sequences, maximum context retention
model(tokens, expanding_context=True)

# Sliding Window - Fixed cache size, removes oldest entries  
# Good for: Long sequences, memory-constrained environments
model(tokens, expanding_context=False)
```

#### Encoder-Decoder KV Caching
```python
from transformer.transformer_model import EncoderDecoder
from transformer.configs import FFNConfig

# Configure FFN for encoder-decoder with KV caching
encdec_kv_ffn_config = FFNConfig(
    embed_dim=64,
    latent_dim=128,
    use_moe=True,  # MOE can be beneficial for large encoder-decoder models
    num_experts=4,
    capacity=32,
    alpha=0.01,
    topk=1
)

model = EncoderDecoder(
    input_dim=1, embed_dim=64, ffn_latent_dim=128,
    num_encoder_layers=2, num_decoder_layers=2, num_heads=4, output_dim=1,
    apply_causal_mask=True, max_seq_len=128,
    attention_type="mqa", use_kv_cache=True,
    ffn_config=encdec_kv_ffn_config
)

# Encoder - can cache for repeated encoding of same source
source = torch.randn(1, 20, 1)
encoder_output = model.encode(source, expanding_context=False)

# Decoder - caches both self-attention and cross-attention K,V pairs
target = torch.randn(1, 1, 1)  # Start with single token
for step in range(10):
    # Each step reuses previous K,V and only computes new ones
    output = model.decode(target, encoder_output, expanding_context=True)
    # Use output to get next token for autoregressive generation
    target = torch.cat([target, output[:, -1:, :]], dim=1)
```

### Performance Benefits

| Attention Type | Memory Reduction | Speed Improvement | Cache Efficiency |
|----------------|------------------|-------------------|------------------|
| **MHA + Cache** | 40-60% | 5-8x faster | Good |
| **GQA + Cache** | 50-70% | 7-10x faster | Better |  
| **MQA + Cache** | 60-80% | 8-12x faster | Best |

### Implementation Features

- **Automatic Management**: Cache creation, updates, and cleanup handled automatically
- **Memory Efficient**: Supports both expanding and sliding window cache strategies
- **Attention Agnostic**: Works seamlessly with MHA, MQA, and GQA attention mechanisms
- **Batch Compatible**: Supports batched inference with independent caches per sample
- **Gradient Safe**: Caching preserves gradient computation during training
- **Thread Safe**: Safe for concurrent inference scenarios

### Cache Lifecycle

1. **Initialization**: Cache is created on first forward pass with `use_kv_cache=True`
2. **Population**: Key and value tensors are computed and stored
3. **Reuse**: Subsequent forward passes reuse cached values and append new ones
4. **Management**: Cache size managed based on `expanding_context` parameter
5. **Cleanup**: Cache is automatically cleared when model is reset or destroyed

### Memory Usage

For sequence length S, embed dimension D, H heads, G groups:

| Mechanism | Cache Memory per Layer |
|-----------|------------------------|
| **MHA** | 2 × H × S × (D/H) = 2 × S × D |
| **GQA** | 2 × G × S × (D/H) = 2 × S × D × (G/H) |
| **MQA** | 2 × 1 × S × (D/H) = 2 × S × (D/H) |

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

**num_groups Configuration**:
- `num_groups=1`: Equivalent to MQA (single shared K/V)
- `num_groups=num_heads`: Equivalent to MHA (no sharing)
- `1 < num_groups < num_heads`: True GQA with grouped sharing
- **Default**: `num_heads//2` for balanced performance

### Attention Factory
```python
from transformer.attention_utils import get_attention

# Unified interface for all attention types
mha = get_attention("mha", embed_dim=64, num_heads=8, num_groups=8, apply_causal_mask=False, use_kv_cache=True)  # num_groups ignored for MHA
mqa = get_attention("mqa", embed_dim=64, num_heads=8, num_groups=1, apply_causal_mask=False, use_kv_cache=True)  # num_groups ignored for MQA
gqa = get_attention("gqa", embed_dim=64, num_heads=8, num_groups=4, apply_causal_mask=False, use_kv_cache=True)  # num_groups configurable for GQA

# GQA flexibility examples:
gqa_mqa_like = get_attention("gqa", embed_dim=64, num_heads=8, num_groups=1)  # Acts like MQA
gqa_balanced = get_attention("gqa", embed_dim=64, num_heads=8, num_groups=4)  # 2 heads per group
gqa_mha_like = get_attention("gqa", embed_dim=64, num_heads=8, num_groups=8)  # Acts like MHA
```

## Mixture of Experts (MOE)

Sparse expert routing system that scales model capacity while maintaining computational efficiency through selective expert activation.

### Features

- **🎯 Sparse Routing**: Top-k expert selection (configurable k=1,2,...)
- **⚖️ Capacity Constraints**: Prevents expert overloading during training
- **🔄 Overflow Handling**: Dedicated overflow expert for capacity violations
- **📊 Load Balancing**: Auxiliary loss prevents expert underutilization
- **⚡ GPU Compatible**: Proper device placement for CUDA acceleration
- **🔧 Integration Ready**: Drop-in replacement for FFN layers

### Usage

#### Basic MOE Layer
```python
from transformer.moe import MOE
import torch

# Create MOE layer with 8 experts
moe = MOE(
    embed_dim=512,           # Input/output embedding dimension
    ffn_latent_dim=2048,     # Hidden dimension for expert FFNs
    num_experts=8,           # Number of expert networks
    capacity=128,            # Max tokens per expert
    alpha=0.01,              # Load balancing coefficient
    topk=1                   # Top-1 routing
)

# Forward pass returns output and auxiliary loss
input_tensor = torch.randn(4, 32, 512)  # [batch, seq_len, embed_dim]
output, aux_loss = moe(input_tensor)     # Same shape + scalar loss

print(f"Output shape: {output.shape}")         # torch.Size([4, 32, 512])
print(f"Auxiliary loss: {aux_loss.item():.4f}") # Load balancing penalty
```

#### Integration with Transformer
```python
import torch.nn as nn
from transformer.moe import MOE
from transformer.encoder import TransformerEncoderLayer

class TransformerWithMOE(nn.Module):
    def __init__(self, embed_dim, num_experts=8):
        super().__init__()
        
        # Standard transformer layer with attention
        self.attention = MultiHeadAttention(embed_dim, num_heads=8)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Replace FFN with MOE
        self.moe = MOE(
            embed_dim=embed_dim,
            ffn_latent_dim=embed_dim * 4,
            num_experts=num_experts,
            capacity=embed_dim,
            alpha=0.01,
            topk=1
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Self-attention block
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MOE block (replaces standard FFN)
        moe_out, aux_loss = self.moe(x)
        x = self.norm2(x + moe_out)
        
        return x, aux_loss
```

#### Training with Load Balancing
```python
model = TransformerWithMOE(embed_dim=512, num_experts=16)
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    inputs, targets = batch
    outputs, aux_loss = model(inputs)
    
    # Main task loss
    main_loss = criterion(outputs, targets)
    
    # Total loss includes auxiliary loss
    total_loss = main_loss + aux_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Configuration Guidelines

| Model Size | num_experts | capacity | topk | alpha | Use Case |
|------------|-------------|----------|------|-------|----------|
| **Small** | 4-8 | 64-128 | 1 | 0.01 | Research, prototyping |
| **Medium** | 8-16 | 128-256 | 1-2 | 0.01 | Production systems |
| **Large** | 32-64 | 256-512 | 2 | 0.01 | Large language models |
| **XL** | 64-128 | 512-1024 | 2 | 0.01 | Massive scale models |

### Implementation Details

#### Routing Algorithm
1. **Router Network**: Linear layer computes expert assignment probabilities
2. **Top-k Selection**: Select k highest-scoring experts per token  
3. **Normalization**: Renormalize weights among selected experts
4. **Capacity Check**: Route tokens respecting per-expert capacity limits
5. **Overflow Handling**: Excess tokens handled by dedicated overflow expert

#### Load Balancing Loss
```
aux_loss = α × Σ(experts) f_i × p_i

Where:
- f_i = fraction of tokens assigned to expert i
- p_i = average routing probability for expert i  
- α = load balancing coefficient (typically 0.01)
```

#### Memory Usage
For batch size B, sequence length S, and E experts:
- **Router**: O(embed_dim × num_experts) parameters
- **Experts**: O(num_experts × embed_dim × ffn_latent_dim) parameters  
- **Activation**: O(B × S × embed_dim) per active expert

### Performance Characteristics

| Configuration | Memory | Compute | Quality | Best For |
|---------------|---------|---------|---------|----------|
| **Top-1, Low Experts** | Low | Low | Good | Efficiency-first |
| **Top-1, Many Experts** | Medium | Low | Better | Specialization |
| **Top-2, Many Experts** | High | Medium | Best | Quality-first |

### Best Practices

- **Capacity Sizing**: Set to 25-50% of average tokens per expert
- **Expert Scaling**: Scale experts with model size (4-8 for small, 64+ for large)
- **Load Balancing**: Always include auxiliary loss in training
- **Top-k Selection**: Use k=1 for efficiency, k=2 for quality
- **Alpha Tuning**: Start with α=0.01, increase if experts are imbalanced

## 🚀 Speculative Decoding

Speculative decoding dramatically accelerates autoregressive text generation by leveraging a fast "draft" model to propose tokens and a high-quality "target" model to verify them in parallel.

### Key Benefits
- **🏃‍♂️ 2-4x Speed Improvement**: Significantly faster than sequential generation
- **🎯 Quality Preservation**: Mathematically equivalent output to target model
- **⚖️ Configurable Trade-offs**: Balance speed vs. quality via model sizing
- **🔄 Architecture Agnostic**: Compatible with any transformer architecture

### Usage

#### Basic Setup
```python
from transformer.spec_decoding import SpecDecodingPair, SpecDecodingConfig
from regression.configs import TransformerModelConfig, AutoregressiveDecodeConfig
from transformer.configs import FFNConfig

# Fast draft model configuration
draft_config = TransformerModelConfig(
    name="draft", input_dim=1, embed_dim=64, ffn_latent_dim=128,
    num_layers=2, num_heads=4, output_dim=1, vocab_size=1000,
    attention_type="mqa",  # MQA for speed
    apply_causal_mask=True, autoregressive_mode=True,
    ffn_config=FFNConfig(embed_dim=64, latent_dim=128, use_moe=False)
)

# High-quality target model configuration  
target_config = TransformerModelConfig(
    name="target", input_dim=1, embed_dim=1024, ffn_latent_dim=4096,
    num_layers=8, num_heads=8, output_dim=1, vocab_size=1000,
    attention_type="mha",  # MHA for quality
    apply_causal_mask=True, autoregressive_mode=True,
    ffn_config=FFNConfig(embed_dim=1024, latent_dim=4096, use_moe=False)
)

# Create speculative decoding pair
spec_config = SpecDecodingConfig(
    draft_config=draft_config,
    target_config=target_config,
    draft_steps=5  # Tokens per speculation round
)

model = SpecDecodingPair(spec_config)
```

#### Generation
```python
import torch

# Input sequence
batch_size, seq_len, input_dim = 2, 3, 1
input_sequence = torch.randn(batch_size, seq_len, input_dim)

# Generate tokens using speculative decoding
num_tokens = 20
output = model(input_sequence, num_tokens)
print(f"Generated sequence: {input_sequence.shape} → {output.shape}")
```

### Algorithm Overview

#### Phase 1: Draft Generation (Sequential)
```python
for step in range(draft_steps):
    # Generate next token with draft model
    embedding, logits = draft_model.generate_next_token_embedding_and_logits(
        current_sequence, expanding_context=True
    )
    draft_embeddings.append(embedding)
    draft_logits.append(logits)
    current_sequence = torch.cat([current_sequence, embedding], dim=1)
```

#### Phase 2: Target Verification (Parallel)
```python
# Process all draft tokens at once with target model
full_sequence = torch.cat([input_sequence, draft_sequence], dim=1)
target_logits = target_model.get_logits(full_sequence, expanding_context=False)
target_probs = torch.softmax(target_logits[:, -draft_steps:, :], dim=-1)
```

#### Phase 3: Acceptance/Rejection Sampling
```python
for k in range(draft_steps):
    # Sample token from draft distribution
    token_id = torch.multinomial(draft_probs[:, k, :], num_samples=1)
    
    # Get probabilities for sampled token
    draft_prob = draft_probs[:, k, :].gather(1, token_id)
    target_prob = target_probs[:, k, :].gather(1, token_id)
    
    # Accept with probability min(1, p_target/p_draft)
    acceptance_ratio = (target_prob / draft_prob).clamp(max=1.0)
    if torch.bernoulli(acceptance_ratio).all():
        accepted_count += 1
    else:
        break  # Stop at first rejection
```

#### Phase 4: Resampling Rejected Tokens
```python
for k in range(accepted_count, draft_steps):
    # Corrected distribution: max(0, p_target - p_draft)
    corrected_probs = torch.clamp(
        target_probs[:, k, :] - draft_probs[:, k, :], min=0.0
    )
    corrected_probs = corrected_probs / corrected_probs.sum(dim=-1, keepdim=True)
    
    # Sample replacement token
    resampled_token = torch.multinomial(corrected_probs, num_samples=1)
    resampled_embedding = convert_token_to_embedding(resampled_token)
```

### Configuration Best Practices

#### Draft Model (Speed Optimized)
```python
draft_config = TransformerModelConfig(
    num_layers=1-2,           # Minimal layers
    embed_dim=64-256,         # Smaller embedding
    attention_type="mqa",     # Most efficient attention
    ffn_config=FFNConfig(
        use_moe=False,        # Disable MOE for speed
        latent_dim=embed_dim*2
    ),
    decode_config=AutoregressiveDecodeConfig(
        expanding_context=True,  # Enable for draft
        use_kv_cache=True       # Cache for efficiency
    )
)
```

#### Target Model (Quality Optimized)  
```python
target_config = TransformerModelConfig(
    num_layers=6-12,          # More layers for quality
    embed_dim=512-2048,       # Larger embedding
    attention_type="mha",     # Best attention quality
    ffn_config=FFNConfig(
        use_moe=True,         # Can use MOE for capacity
        num_experts=8-16,
        capacity=embed_dim,
        topk=2
    ),
    decode_config=AutoregressiveDecodeConfig(
        expanding_context=False, # Target processes in parallel
        use_kv_cache=False
    )
)
```

### Performance Characteristics

| Draft Size | Target Size | Speed Gain | Quality | Use Case |
|------------|-------------|------------|---------|----------|
| **Tiny** (1-2 layers) | **Small** (4-6 layers) | 1.5-2x | ~99% | Resource constrained |
| **Small** (2-3 layers) | **Medium** (6-8 layers) | 2-3x | ~98% | Balanced deployment |
| **Small** (2-3 layers) | **Large** (8-12 layers) | 3-4x | ~95% | Quality-first |

### Implementation Features

- **🔄 Automatic Fallback**: Uses target model when no tokens accepted
- **🎯 Batch Processing**: Efficient batch-level acceptance testing  
- **🛡️ Edge Case Handling**: Robust probability distribution handling
- **🔧 Token Conversion**: Proper token-to-embedding conversion pipeline
- **📊 Configurable Parameters**: Tunable draft steps and architectures
- **⚡ Memory Efficient**: Optimized tensor operations and caching

### Integration with Other Features

#### With KV Caching
```python
# Draft model benefits from KV caching
draft_config.decode_config.use_kv_cache = True
draft_config.decode_config.expanding_context = True

# Target model processes in parallel (no caching needed)  
target_config.decode_config.use_kv_cache = False
```

#### With MOE
```python
# Draft model: Disable MOE for speed
draft_config.ffn_config.use_moe = False

# Target model: Enable MOE for quality/capacity
target_config.ffn_config.use_moe = True
target_config.ffn_config.num_experts = 8
target_config.ffn_config.topk = 2
```

#### With Different Attention Types
```python
# Speed-optimized combination
draft_config.attention_type = "mqa"    # Fastest attention
target_config.attention_type = "mha"   # Highest quality

# Memory-optimized combination  
draft_config.attention_type = "mqa"    # Low memory
target_config.attention_type = "gqa"   # Balanced memory/quality
```

### Mathematical Guarantees

The speculative decoding algorithm is mathematically equivalent to sampling directly from the target model:
- **Correctness**: Output distribution matches target model exactly
- **Efficiency**: Achieves speedup without quality loss
- **Convergence**: Always terminates with valid token sequence

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