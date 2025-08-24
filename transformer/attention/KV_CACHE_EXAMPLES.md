# KV Cache Usage Examples

Comprehensive examples demonstrating key-value caching across all attention mechanisms for optimized autoregressive inference.

## Table of Contents

- [Basic KV Caching](#basic-kv-caching)
- [Autoregressive Generation](#autoregressive-generation) 
- [Encoder-Decoder with Caching](#encoder-decoder-with-caching)
- [Performance Benchmarks](#performance-benchmarks)
- [Memory Management](#memory-management)
- [Production Usage Patterns](#production-usage-patterns)

## Basic KV Caching

### Multi-Head Attention (MHA)

```python
import torch
from transformer.attention.mha import MultiHeadAttention

# Initialize MHA with KV caching enabled
mha = MultiHeadAttention(
    embed_dim=512,
    num_heads=8,
    apply_causal_mask=True,
    use_kv_cache=True
)

# First forward pass - initializes cache
batch_size, seq_len, embed_dim = 1, 10, 512
initial_input = torch.randn(batch_size, seq_len, embed_dim)

print(f"Before first pass: cache is {mha.kv_cache}")
output1 = mha(initial_input, kv=None, expanding_context=True)
print(f"After first pass: cache contains keys/values for {mha.kv_cache['key'].shape[2]} tokens")

# Subsequent pass - reuses cache
next_token = torch.randn(batch_size, 1, embed_dim)
output2 = mha(next_token, kv=None, expanding_context=True)
print(f"After second pass: cache contains keys/values for {mha.kv_cache['key'].shape[2]} tokens")

# Output shapes
print(f"First output shape: {output1.shape}")    # [1, 10, 512]
print(f"Second output shape: {output2.shape}")   # [1, 1, 512]
```

### Multi-Query Attention (MQA) - Most Memory Efficient

```python
from transformer.attention.mqa import MultiQueryAttention

# MQA uses single K,V heads - smallest cache footprint
mqa = MultiQueryAttention(
    embed_dim=512,
    num_heads=8,  # 8 query heads, but only 1 key and 1 value head
    apply_causal_mask=True,
    use_kv_cache=True
)

initial_input = torch.randn(1, 10, 512)
output1 = mqa(initial_input, kv=None, expanding_context=True)

# Cache size comparison: MQA cache is 8x smaller than MHA
print(f"MQA cache key shape: {mqa.kv_cache['key'].shape}")      # [1, 1, 10, 64]
print(f"Memory per token: {mqa.kv_cache['key'].numel() * 4} bytes")  # Much smaller

next_token = torch.randn(1, 1, 512)
output2 = mqa(next_token, kv=None, expanding_context=True)
print(f"Cache after second pass: {mqa.kv_cache['key'].shape[2]} tokens")
```

### Group Query Attention (GQA) - Balanced Approach

```python
from transformer.attention.gqa import GroupQueryAttention

# GQA balances between MHA and MQA
gqa = GroupQueryAttention(
    embed_dim=512,
    num_heads=8,
    num_groups=4,  # 4 K,V heads shared by 8 Q heads (2 queries per group)
    apply_causal_mask=True,
    use_kv_cache=True
)

initial_input = torch.randn(1, 10, 512)
output1 = gqa(initial_input, kv=None, expanding_context=True)

# Cache size: 4 groups vs 8 heads (MHA) vs 1 head (MQA)
print(f"GQA cache key shape: {gqa.kv_cache['key'].shape}")      # [1, 4, 10, 64]
print(f"Cache efficiency: 50% of MHA size, 4x larger than MQA")

next_token = torch.randn(1, 1, 512)
output2 = gqa(next_token, kv=None, expanding_context=True)
```

## Autoregressive Generation

### Token-by-Token Generation Loop

```python
import torch
from transformer.attention.gqa import GroupQueryAttention

def autoregressive_generation(attention_module, initial_tokens, max_length=20):
    """Simulate autoregressive text generation with KV caching."""
    
    # Prepare initial sequence
    current_sequence = initial_tokens.clone()
    generated_tokens = []
    
    print(f"Starting generation with {current_sequence.shape[1]} tokens")
    
    for step in range(max_length):
        if step == 0:
            # First pass: process entire sequence, initialize cache
            input_tokens = current_sequence
            expanding_context = True
            print(f"Step {step}: Processing {input_tokens.shape[1]} tokens (cache init)")
        else:
            # Subsequent passes: only process new token, reuse cache
            input_tokens = current_sequence[:, -1:, :]  # Just the last token
            expanding_context = True
            print(f"Step {step}: Processing 1 token (cache reuse)")
        
        # Forward pass - cache automatically managed
        with torch.no_grad():
            output = attention_module(input_tokens, kv=None, expanding_context=expanding_context)
        
        # Simulate next token selection (in practice, this would be from a language model head)
        next_token_embedding = torch.randn(1, 1, 512)  # Simulate sampling
        current_sequence = torch.cat([current_sequence, next_token_embedding], dim=1)
        generated_tokens.append(next_token_embedding)
        
        # Show cache growth
        cache_size = attention_module.kv_cache['key'].shape[2]
        print(f"  Cache now holds {cache_size} tokens")
        
        # Early stopping condition (simulate end-of-sequence)
        if step > 5 and torch.rand(1) < 0.3:
            print(f"Generation complete at step {step}")
            break
    
    return current_sequence, generated_tokens

# Example usage
gqa = GroupQueryAttention(embed_dim=512, num_heads=8, num_groups=4, 
                          apply_causal_mask=True, use_kv_cache=True)

initial_prompt = torch.randn(1, 5, 512)  # Start with 5 tokens
final_sequence, generated = autoregressive_generation(gqa, initial_prompt, max_length=15)

print(f"\nGeneration Summary:")
print(f"Initial tokens: {initial_prompt.shape[1]}")
print(f"Generated tokens: {len(generated)}")
print(f"Final sequence length: {final_sequence.shape[1]}")
```

### Context Management Strategies

```python
import torch
from transformer.attention.mha import MultiHeadAttention

def compare_context_strategies():
    """Compare expanding vs sliding window context management."""
    
    mha_expanding = MultiHeadAttention(embed_dim=256, num_heads=4, 
                                      apply_causal_mask=True, use_kv_cache=True)
    mha_sliding = MultiHeadAttention(embed_dim=256, num_heads=4, 
                                    apply_causal_mask=True, use_kv_cache=True)
    
    # Initial sequence
    initial_tokens = torch.randn(1, 8, 256)
    
    print("=== Expanding Context Strategy ===")
    # Process initial sequence
    _ = mha_expanding(initial_tokens, kv=None, expanding_context=True)
    print(f"After initial: cache size = {mha_expanding.kv_cache['key'].shape[2]}")
    
    # Add tokens one by one - cache grows
    for i in range(5):
        new_token = torch.randn(1, 1, 256)
        _ = mha_expanding(new_token, kv=None, expanding_context=True)
        cache_size = mha_expanding.kv_cache['key'].shape[2]
        print(f"Step {i+1}: cache size = {cache_size} (grew by 1)")
    
    print("\n=== Sliding Window Strategy ===")
    window_size = 8  # Fixed context window
    
    # Process initial sequence  
    _ = mha_sliding(initial_tokens, kv=None, expanding_context=False)
    print(f"After initial: cache size = {mha_sliding.kv_cache['key'].shape[2]}")
    
    # Add tokens - cache size stays fixed (sliding window)
    for i in range(5):
        new_token = torch.randn(1, 1, 256)
        _ = mha_sliding(new_token, kv=None, expanding_context=False)
        cache_size = mha_sliding.kv_cache['key'].shape[2]
        print(f"Step {i+1}: cache size = {cache_size} (sliding window)")

compare_context_strategies()
```

## Encoder-Decoder with Caching

```python
import torch
from transformer.attention.mha import MultiHeadAttention

def encoder_decoder_caching_example():
    """Demonstrate KV caching in encoder-decoder architecture."""
    
    # Encoder self-attention (can cache for repeated encoding)
    encoder_self_attn = MultiHeadAttention(embed_dim=512, num_heads=8, 
                                          apply_causal_mask=False, use_kv_cache=True)
    
    # Decoder self-attention (caches for autoregressive generation)
    decoder_self_attn = MultiHeadAttention(embed_dim=512, num_heads=8, 
                                          apply_causal_mask=True, use_kv_cache=True)
    
    # Decoder cross-attention (caches encoder outputs as K,V)
    decoder_cross_attn = MultiHeadAttention(embed_dim=512, num_heads=8, 
                                           apply_causal_mask=False, use_kv_cache=True)
    
    # Source sequence (encoder input)
    source_sequence = torch.randn(1, 15, 512)  # [batch, src_len, embed_dim]
    
    print("=== Encoder Phase ===")
    # Encode source - could be cached if same source used multiple times
    encoder_output = encoder_self_attn(source_sequence, kv=None, expanding_context=False)
    print(f"Encoded source: {encoder_output.shape}")
    print(f"Encoder cache size: {encoder_self_attn.kv_cache['key'].shape[2]} tokens")
    
    print("\n=== Decoder Phase (Autoregressive) ===")
    # Decoder autoregressive generation
    decoder_sequence = torch.randn(1, 1, 512)  # Start with <START> token
    
    for step in range(8):
        print(f"\nDecoder Step {step + 1}:")
        
        # Self-attention in decoder (causal, uses cache)
        if step == 0:
            # First step: process initial token, initialize cache
            decoder_self_out = decoder_self_attn(decoder_sequence, kv=None, expanding_context=True)
            print(f"  Self-attention cache initialized with {decoder_self_attn.kv_cache['key'].shape[2]} tokens")
        else:
            # Subsequent steps: process only new token, reuse cache
            new_token = decoder_sequence[:, -1:, :]
            decoder_self_out = decoder_self_attn(new_token, kv=None, expanding_context=True)
            print(f"  Self-attention cache now has {decoder_self_attn.kv_cache['key'].shape[2]} tokens")
        
        # Cross-attention: decoder queries attend to encoder output
        if step == 0:
            # Initialize cross-attention cache with encoder outputs as K,V
            decoder_input_for_cross = decoder_self_out
        else:
            # Use only the output for the new token
            decoder_input_for_cross = decoder_self_out[:, -1:, :]
        
        cross_attn_out = decoder_cross_attn(decoder_input_for_cross, 
                                          kv=encoder_output, 
                                          expanding_context=True)
        print(f"  Cross-attention: query shape {decoder_input_for_cross.shape}, KV from encoder")
        
        # Simulate adding new token to decoder sequence
        next_token = torch.randn(1, 1, 512)
        decoder_sequence = torch.cat([decoder_sequence, next_token], dim=1)
        print(f"  Decoder sequence now: {decoder_sequence.shape[1]} tokens")

encoder_decoder_caching_example()
```

## Performance Benchmarks

```python
import torch
import time
from transformer.attention.mha import MultiHeadAttention
from transformer.attention.mqa import MultiQueryAttention
from transformer.attention.gqa import GroupQueryAttention

def benchmark_kv_caching():
    """Benchmark the performance impact of KV caching."""
    
    def time_generation(attention_module, use_cache, num_steps=50):
        """Time autoregressive generation with/without caching."""
        
        # Reset cache
        if hasattr(attention_module, 'kv_cache'):
            attention_module.kv_cache = None
        
        initial_seq = torch.randn(1, 10, 512)
        
        start_time = time.time()
        
        if use_cache:
            # First pass - initialize cache
            _ = attention_module(initial_seq, kv=None, expanding_context=True)
            
            # Subsequent passes - use cache
            for _ in range(num_steps):
                next_token = torch.randn(1, 1, 512)
                _ = attention_module(next_token, kv=None, expanding_context=True)
        else:
            # No caching - recompute everything each time
            current_seq = initial_seq
            for _ in range(num_steps):
                next_token = torch.randn(1, 1, 512)
                current_seq = torch.cat([current_seq, next_token], dim=1)
                _ = attention_module(current_seq, kv=None, expanding_context=False)
        
        return time.time() - start_time
    
    # Test different attention mechanisms
    attention_configs = [
        ("MHA", MultiHeadAttention(512, 8, True, True)),
        ("GQA", GroupQueryAttention(512, 8, 4, True, True)), 
        ("MQA", MultiQueryAttention(512, 8, True, True))
    ]
    
    print("Performance Benchmark (50 generation steps):")
    print("-" * 60)
    print(f"{'Attention':<8} {'No Cache':<12} {'With Cache':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for name, attention in attention_configs:
        # Warmup
        _ = time_generation(attention, use_cache=True, num_steps=5)
        _ = time_generation(attention, use_cache=False, num_steps=5)
        
        # Benchmark
        time_no_cache = time_generation(attention, use_cache=False, num_steps=50)
        time_with_cache = time_generation(attention, use_cache=True, num_steps=50)
        
        speedup = time_no_cache / time_with_cache
        
        print(f"{name:<8} {time_no_cache:<12.4f} {time_with_cache:<12.4f} {speedup:<10.2f}x")

# Run benchmark
with torch.no_grad():
    benchmark_kv_caching()
```

## Memory Management

```python
import torch
from transformer.attention.mha import MultiHeadAttention

def memory_usage_analysis():
    """Analyze memory usage patterns with KV caching."""
    
    def get_cache_memory_mb(attention_module):
        """Calculate cache memory usage in MB."""
        if attention_module.kv_cache is None:
            return 0.0
        
        total_elements = 0
        for tensor in attention_module.kv_cache.values():
            total_elements += tensor.numel()
        
        # Assume float32 (4 bytes per element)
        return total_elements * 4 / (1024 * 1024)
    
    # Different configurations
    configs = [
        ("MHA-8heads", MultiHeadAttention(512, 8, True, True)),
        ("MHA-16heads", MultiHeadAttention(512, 16, True, True)),
        ("GQA-8h-4g", GroupQueryAttention(512, 8, 4, True, True)),
        ("GQA-8h-2g", GroupQueryAttention(512, 8, 2, True, True)),
        ("MQA-8heads", MultiQueryAttention(512, 8, True, True)),
    ]
    
    print("Memory Usage Analysis:")
    print("-" * 70)
    print(f"{'Config':<12} {'Seq=50':<10} {'Seq=100':<10} {'Seq=200':<10} {'Seq=500':<10}")
    print("-" * 70)
    
    for name, attention in configs:
        memory_usage = []
        
        for seq_len in [50, 100, 200, 500]:
            # Initialize cache with sequence
            attention.kv_cache = None  # Reset
            initial_seq = torch.randn(1, seq_len, 512)
            _ = attention(initial_seq, kv=None, expanding_context=True)
            
            memory_mb = get_cache_memory_mb(attention)
            memory_usage.append(f"{memory_mb:.1f}MB")
        
        print(f"{name:<12} {' '<10} {' ':<10} {' ':<10} {' ':<10}")
        
    print("-" * 70)
    print("Note: Memory usage scales linearly with sequence length")
    print("MQA < GQA < MHA in terms of cache memory requirements")

memory_usage_analysis()
```

## Production Usage Patterns

### Pattern 1: Chat/Conversation System

```python
import torch
from transformer.attention.gqa import GroupQueryAttention

class ConversationSystem:
    """Example chat system using KV caching for efficient response generation."""
    
    def __init__(self):
        self.attention = GroupQueryAttention(
            embed_dim=1024, num_heads=16, num_groups=8,
            apply_causal_mask=True, use_kv_cache=True
        )
        self.conversation_history = []
    
    def add_message(self, message_embedding):
        """Add a message to conversation history."""
        self.conversation_history.append(message_embedding)
        
        if len(self.conversation_history) == 1:
            # First message - initialize cache
            full_history = message_embedding
            _ = self.attention(full_history, kv=None, expanding_context=True)
        else:
            # Subsequent messages - use cache efficiently
            _ = self.attention(message_embedding, kv=None, expanding_context=True)
    
    def generate_response(self, max_tokens=50):
        """Generate response using cached conversation context."""
        response_tokens = []
        
        for _ in range(max_tokens):
            # Generate next token (in practice, this would use a language model head)
            next_token = torch.randn(1, 1, 1024)  # Simulated token
            
            # Process with cached context - very fast
            _ = self.attention(next_token, kv=None, expanding_context=True)
            response_tokens.append(next_token)
            
            # Stop condition (simulate end-of-response token)
            if torch.rand(1) < 0.1:
                break
        
        return torch.cat(response_tokens, dim=1)
    
    def reset_conversation(self):
        """Clear conversation and cache."""
        self.conversation_history = []
        self.attention.kv_cache = None

# Usage example
chat_system = ConversationSystem()

# User message 1
user_msg_1 = torch.randn(1, 12, 1024)  # 12 tokens
chat_system.add_message(user_msg_1)
response_1 = chat_system.generate_response(max_tokens=20)
print(f"Response 1: {response_1.shape}")

# User message 2 - cache reused efficiently
user_msg_2 = torch.randn(1, 8, 1024)   # 8 tokens  
chat_system.add_message(user_msg_2)
response_2 = chat_system.generate_response(max_tokens=15)
print(f"Response 2: {response_2.shape}")
```

### Pattern 2: Batch Inference with Different Sequence Lengths

```python
import torch
from transformer.attention.mqa import MultiQueryAttention

def batch_inference_example():
    """Handle batch inference with KV caching (simplified example)."""
    
    # Note: In practice, each sample in batch would have separate cache
    # This example shows the concept
    
    mqa = MultiQueryAttention(embed_dim=512, num_heads=8, apply_causal_mask=True, use_kv_cache=True)
    
    # Simulate different samples at different generation steps
    samples = [
        {"id": "sample_1", "current_length": 5, "max_length": 20},
        {"id": "sample_2", "current_length": 10, "max_length": 25}, 
        {"id": "sample_3", "current_length": 3, "max_length": 15},
    ]
    
    print("Batch Inference Simulation:")
    print("-" * 40)
    
    for step in range(10):
        print(f"\nGeneration Step {step + 1}:")
        
        for sample in samples:
            if sample["current_length"] >= sample["max_length"]:
                print(f"  {sample['id']}: Complete")
                continue
            
            if step == 0:
                # First step: initialize with existing sequence
                initial_tokens = torch.randn(1, sample["current_length"], 512)
                _ = mqa(initial_tokens, kv=None, expanding_context=True)
                print(f"  {sample['id']}: Initialized cache with {sample['current_length']} tokens")
            else:
                # Subsequent steps: add one token
                new_token = torch.randn(1, 1, 512)
                _ = mqa(new_token, kv=None, expanding_context=True)
                sample["current_length"] += 1
                print(f"  {sample['id']}: Generated token, length now {sample['current_length']}")
            
            # In practice, each sample would have its own cache/attention instance

batch_inference_example()
```

### Pattern 3: Long Document Processing with Sliding Window

```python
import torch
from transformer.attention.mha import MultiHeadAttention

def long_document_processing():
    """Process long documents using sliding window KV caching."""
    
    attention = MultiHeadAttention(embed_dim=512, num_heads=8, 
                                  apply_causal_mask=False, use_kv_cache=True)
    
    # Simulate long document as chunks
    document_chunks = [
        torch.randn(1, 100, 512),  # Chunk 1: 100 tokens
        torch.randn(1, 100, 512),  # Chunk 2: 100 tokens  
        torch.randn(1, 100, 512),  # Chunk 3: 100 tokens
        torch.randn(1, 100, 512),  # Chunk 4: 100 tokens
        torch.randn(1, 100, 512),  # Chunk 5: 100 tokens
    ]
    
    max_context_length = 200  # Sliding window size
    
    print("Long Document Processing with Sliding Window:")
    print("-" * 50)
    
    for i, chunk in enumerate(document_chunks):
        print(f"\nProcessing Chunk {i + 1} ({chunk.shape[1]} tokens):")
        
        if i == 0:
            # First chunk - initialize cache
            _ = attention(chunk, kv=None, expanding_context=False)
            current_context_size = chunk.shape[1]
        else:
            # Subsequent chunks - add to cache with sliding window
            _ = attention(chunk, kv=None, expanding_context=False)
            current_context_size = min(current_context_size + chunk.shape[1], max_context_length)
        
        cache_size = attention.kv_cache['key'].shape[2]
        print(f"  Cache size: {cache_size} tokens (max: {max_context_length})")
        print(f"  Effective context: {current_context_size} tokens")
        
        if cache_size >= max_context_length:
            print(f"  Sliding window active - maintaining fixed context size")

long_document_processing()
```

## Best Practices Summary

1. **Choose Attention Type Based on Requirements**:
   - MQA: Maximum speed, minimum memory, slight quality trade-off
   - GQA: Balanced speed/memory/quality for production
   - MHA: Maximum quality, higher memory/compute cost

2. **Context Management**:
   - Use `expanding_context=True` for short sequences or chat
   - Use `expanding_context=False` for long documents or memory-constrained environments

3. **Memory Optimization**:
   - Monitor cache size growth in long-running applications
   - Reset cache periodically for memory management
   - Consider gradient checkpointing if training with caching

4. **Performance Tips**:
   - Pre-warm caches before benchmarking
   - Use appropriate batch sizes to balance memory and throughput
   - Profile memory usage in production to avoid OOM errors

5. **Integration**:
   - KV caching works seamlessly with all transformer architectures
   - Can be enabled/disabled at model configuration time
   - Preserves gradient computation for training scenarios