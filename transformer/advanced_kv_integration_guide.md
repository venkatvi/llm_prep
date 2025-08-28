# Advanced KV Cache Integration Guide

## 🎯 Overview

This guide demonstrates how to integrate the advanced KV cache system into your existing transformer architecture and speculative decoding implementation.

## 🏗️ Architecture Blueprint

```
┌─────────────────────────────────────────────────────────────┐
│                  Advanced KV Cache System                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Memory Pool │  │ Eviction    │  │ Cache Stats │          │
│  │ Management  │  │ Policies    │  │ & Monitor   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Per-Layer KV Caches                         │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │ │
│  │  │ Layer 0  │ │ Layer 1  │ │ Layer 2  │ │   ...    │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Per-Sequence Cache Entries                   │ │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │ │
│  │ │ seq_1   │ │ seq_2   │ │ batch_0 │ │   ...   │       │ │
│  │ │ K:[...] │ │ K:[...] │ │ K:[...] │ │ K:[...] │       │ │
│  │ │ V:[...] │ │ V:[...] │ │ V:[...] │ │ V:[...] │       │ │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Integration Steps

### Step 1: Update Attention Mechanisms

```python
# transformer/attention/mha_v2.py
from transformer.kv_cache_v2 import CachedAttentionMixin, AdvancedKVCache

class MultiHeadAttentionV2(CachedAttentionMixin, nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, layer_id: int = 0, **kwargs):
        super().__init__()
        self.layer_id = layer_id
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Existing attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: Optional[torch.Tensor] = None, 
        value: Optional[torch.Tensor] = None,
        expanding_context: bool = False,
        sequence_id: str = "default"
    ) -> torch.Tensor:
        
        batch_size, seq_len, _ = query.shape
        
        # Self-attention case
        if key is None:
            key = value = query
        
        # Project inputs
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if expanding_context and self.kv_cache:
            # Try to get cached K, V
            cached_kv = self._get_cached_kv(self.layer_id, seq_len)
            
            if cached_kv is not None:
                # Cache hit - use cached K, V and only compute for new tokens
                cached_k, cached_v = cached_kv
                
                # Reshape cached tensors
                k_cached = cached_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                v_cached = cached_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
                # Compute only new K, V if sequence extended
                if seq_len > k_cached.size(2):
                    new_tokens = key[:, k_cached.size(2):, :]
                    k_new = self.k_proj(new_tokens).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                    v_new = self.v_proj(new_tokens).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                    
                    k = torch.cat([k_cached, k_new], dim=2)
                    v = torch.cat([v_cached, v_new], dim=2)
                    
                    # Update cache with extended K, V
                    k_flat = k.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
                    v_flat = v.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
                    self._cache_kv(self.layer_id, k_flat, v_flat)
                else:
                    k, v = k_cached, v_cached
            else:
                # Cache miss - compute full K, V
                k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                # Cache the computed K, V
                k_flat = k.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                v_flat = v.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                self._cache_kv(self.layer_id, k_flat, v_flat)
        else:
            # No caching - compute K, V normally
            k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)
```

### Step 2: Update Transformer Models

```python
# transformer/transformer_model_v2.py
from transformer.kv_cache_v2 import AdvancedKVCache, CacheConfig

class AutoregressiveTransformerModelV2(nn.Module):
    def __init__(self, config: TransformerModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize advanced KV cache
        self.cache_config = CacheConfig(
            max_batch_size=config.decode_config.max_batch_size if hasattr(config.decode_config, 'max_batch_size') else 32,
            max_sequence_length=config.max_seq_len,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            eviction_policy=EvictionPolicy.SLIDING_WINDOW,
            sliding_window_size=config.max_seq_len // 2,
            memory_limit_mb=200,  # Configurable based on available memory
            preallocate_memory=True,
        )
        
        self.kv_cache = AdvancedKVCache(self.cache_config) if config.decode_config.use_kv_cache else None
        
        # Model layers with cache-aware attention
        self.layers = nn.ModuleList([
            TransformerLayerV2(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                layer_id=i,
                kv_cache=self.kv_cache
            ) for i in range(config.num_layers)
        ])
    
    def forward(self, x: torch.Tensor, expanding_context: bool = False, sequence_id: str = "default") -> torch.Tensor:
        # Set sequence ID for all layers
        if self.kv_cache:
            for layer in self.layers:
                layer.attention.set_cache(self.kv_cache, sequence_id)
        
        # Forward pass through layers
        for layer in self.layers:
            x = layer(x, expanding_context=expanding_context)
        
        return x
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        if self.kv_cache:
            return self.kv_cache.get_global_stats()
        return {}
    
    def clear_cache(self, sequence_id: Optional[str] = None):
        """Clear cache for specific sequence or all sequences."""
        if self.kv_cache:
            if sequence_id:
                self.kv_cache.clear_sequence(sequence_id)
            else:
                self.kv_cache.clear_all()
```

### Step 3: Enhanced Speculative Decoding

```python
# transformer/spec_decoding_v2.py
from transformer.kv_cache_v2 import create_cache_config_for_speculative_decoding

class SpecDecodingPairV2(nn.Module):
    def __init__(self, spec_config: SpecDecodingConfig):
        super().__init__()
        self.config = spec_config
        
        # Create optimized cache configurations
        draft_cache_config, target_cache_config = create_cache_config_for_speculative_decoding()
        
        # Initialize models with advanced caching
        self.draft_model = AutoregressiveTransformerModelV2(spec_config.draft_config)
        self.target_model = AutoregressiveTransformerModelV2(spec_config.target_config)
        
        # Configure caches for speculative decoding patterns
        if self.draft_model.kv_cache:
            # Draft model benefits from aggressive caching
            self.draft_model.kv_cache.config.eviction_policy = EvictionPolicy.SLIDING_WINDOW
            self.draft_model.kv_cache.config.sliding_window_size = 256
        
        if self.target_model.kv_cache:
            # Target model uses minimal caching (or none)
            self.target_model.kv_cache.config.eviction_policy = EvictionPolicy.NONE
    
    def forward(self, x: torch.Tensor, n: int, batch_sequence_ids: Optional[List[str]] = None) -> torch.Tensor:
        """Enhanced forward pass with batch sequence tracking."""
        current_output = x.clone()
        batch_size = x.size(0)
        
        # Generate unique sequence IDs for batch processing
        if batch_sequence_ids is None:
            batch_sequence_ids = [f"batch_{i}_{int(time.time())}" for i in range(batch_size)]
        
        iteration = 0
        while (current_output.size(1) - x.size(1)) < n:
            print(f"Speculative decoding iteration: {iteration}")
            
            # Phase 1: Draft generation with per-sequence caching
            draft_sequences = []
            draft_probs_list = []
            
            for i, seq_id in enumerate(batch_sequence_ids):
                single_input = current_output[i:i+1]  # Single sequence
                draft_seq, draft_probs = self.generate_draft_sequence(single_input, seq_id)
                draft_sequences.append(draft_seq)
                draft_probs_list.append(draft_probs)
            
            # Combine batch results
            draft_sequence = torch.cat(draft_sequences, dim=0)
            draft_probs = torch.cat(draft_probs_list, dim=0)
            
            # Phase 2: Target verification (parallel processing, minimal caching)
            full_sequence = torch.cat([current_output, draft_sequence], dim=1)
            target_probs = self.verify_draft_with_target(full_sequence, "target_verification")
            
            # Phase 3 & 4: Acceptance and resampling
            final_count, resampled_embeddings = self.process_acceptance_and_resampling(
                draft_probs, target_probs, draft_sequence
            )
            
            # Update current output
            current_output = self.assemble_final_sequence(
                current_output, draft_sequence, final_count, resampled_embeddings
            )
            
            iteration += 1
            
            # Periodically optimize cache memory
            if iteration % 5 == 0:
                self.optimize_cache_memory()
        
        return current_output
    
    def generate_draft_sequence(self, x: torch.Tensor, sequence_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft sequence with sequence-specific caching."""
        generated_embeddings = []
        generated_logits = []
        current_sequence = x.clone()
        
        for step in range(self.config.draft_steps):
            # Use sequence-specific caching
            embedding, logits = self.draft_model.generate_next_token_embedding_and_logits(
                current_sequence, expanding_context=True, sequence_id=sequence_id
            )
            generated_embeddings.append(embedding)
            generated_logits.append(logits)
            current_sequence = torch.cat([current_sequence, embedding], dim=1)
        
        draft_sequence = torch.cat(generated_embeddings, dim=1)
        draft_logits = torch.cat(generated_logits, dim=1)
        draft_probs = torch.softmax(draft_logits, dim=-1)
        
        return draft_sequence, draft_probs
    
    def optimize_cache_memory(self):
        """Trigger memory optimization across both models."""
        if self.draft_model.kv_cache:
            self.draft_model.kv_cache.optimize_memory()
        if self.target_model.kv_cache:
            self.target_model.kv_cache.optimize_memory()
    
    def get_comprehensive_stats(self) -> Dict:
        """Get detailed statistics for both models."""
        return {
            "draft_model_cache": self.draft_model.get_cache_stats(),
            "target_model_cache": self.target_model.get_cache_stats(),
        }
```

## 📊 Performance Monitoring

### Cache Performance Metrics

```python
class CachePerformanceMonitor:
    def __init__(self, cache: AdvancedKVCache):
        self.cache = cache
        self.metrics_history = []
    
    def collect_metrics(self) -> Dict:
        """Collect comprehensive performance metrics."""
        stats = self.cache.get_global_stats()
        
        metrics = {
            "timestamp": time.time(),
            "hit_rate": stats["global_hit_rate"],
            "memory_usage_mb": stats["total_memory_mb"],
            "evictions": stats["total_evictions"],
            "memory_pool_efficiency": stats["memory_pool"]["pool_hit_rate"],
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def analyze_performance(self) -> Dict:
        """Analyze cache performance trends."""
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}
        
        recent = self.metrics_history[-10:]  # Last 10 measurements
        
        avg_hit_rate = sum(m["hit_rate"] for m in recent) / len(recent)
        avg_memory = sum(m["memory_usage_mb"] for m in recent) / len(recent)
        
        hit_rate_trend = "stable"
        if len(recent) >= 3:
            if recent[-1]["hit_rate"] > recent[-3]["hit_rate"] + 0.1:
                hit_rate_trend = "improving"
            elif recent[-1]["hit_rate"] < recent[-3]["hit_rate"] - 0.1:
                hit_rate_trend = "degrading"
        
        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_memory_usage": avg_memory,
            "hit_rate_trend": hit_rate_trend,
            "recommendations": self._generate_recommendations(avg_hit_rate, avg_memory)
        }
    
    def _generate_recommendations(self, hit_rate: float, memory_usage: float) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if hit_rate < 0.5:
            recommendations.append("Consider increasing cache size or adjusting eviction policy")
        
        if memory_usage > 500:  # MB
            recommendations.append("High memory usage - consider more aggressive eviction")
        
        if hit_rate > 0.9 and memory_usage < 100:
            recommendations.append("Excellent performance - consider reducing cache size to save memory")
        
        return recommendations
```

## 🎛️ Configuration Examples

### Production Configurations

```python
# For large language models (LLM inference)
llm_config = CacheConfig(
    max_batch_size=64,
    max_sequence_length=4096,
    embed_dim=4096,
    num_heads=32,
    num_layers=48,
    eviction_policy=EvictionPolicy.ADAPTIVE,
    memory_limit_mb=2048,  # 2GB cache limit
    preallocate_memory=True,
    use_memory_pool=True,
)

# For mobile/edge deployment
mobile_config = CacheConfig(
    max_batch_size=4,
    max_sequence_length=512,
    embed_dim=256,
    num_heads=8,
    num_layers=12,
    eviction_policy=EvictionPolicy.LRU,
    memory_limit_mb=64,  # 64MB limit
    preallocate_memory=False,
    use_memory_pool=True,
)

# For research/experimentation
research_config = CacheConfig(
    max_batch_size=16,
    max_sequence_length=1024,
    embed_dim=512,
    num_heads=16,
    num_layers=24,
    eviction_policy=EvictionPolicy.NONE,  # No eviction for reproducibility
    memory_limit_mb=512,
    preallocate_memory=True,
    use_memory_pool=False,  # Disable pooling for debugging
)
```

## 🧪 Testing and Validation

```python
def test_advanced_cache_integration():
    """Test the advanced cache integration."""
    config = CacheConfig(
        max_batch_size=4,
        max_sequence_length=128,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        eviction_policy=EvictionPolicy.SLIDING_WINDOW,
        memory_limit_mb=50
    )
    
    cache = AdvancedKVCache(config)
    monitor = CachePerformanceMonitor(cache)
    
    # Simulate realistic workload
    for iteration in range(100):
        # Simulate variable sequence lengths
        seq_len = random.randint(16, 128)
        sequence_id = f"seq_{iteration % 10}"  # 10 different sequences
        
        for layer_id in range(config.num_layers):
            key = torch.randn(seq_len, config.embed_dim)
            value = torch.randn(seq_len, config.embed_dim)
            
            # Test cache operations
            cache.set_cache(layer_id, sequence_id, key, value)
            cached_kv = cache.get_cache(layer_id, sequence_id, seq_len)
            
            assert cached_kv is not None, f"Cache miss unexpected for {sequence_id}"
        
        # Collect metrics
        if iteration % 10 == 0:
            metrics = monitor.collect_metrics()
            print(f"Iteration {iteration}: Hit Rate = {metrics['hit_rate']:.2%}")
    
    # Final analysis
    analysis = monitor.analyze_performance()
    print("\nFinal Performance Analysis:")
    print(f"Average Hit Rate: {analysis['avg_hit_rate']:.2%}")
    print(f"Average Memory Usage: {analysis['avg_memory_usage']:.1f} MB")
    print(f"Trend: {analysis['hit_rate_trend']}")
    print("Recommendations:", analysis['recommendations'])

if __name__ == "__main__":
    test_advanced_cache_integration()
```

## 🚀 Migration Guide

### From Basic KV Cache to Advanced Cache

1. **Replace cache initialization:**
   ```python
   # Old
   self.use_kv_cache = config.use_kv_cache
   self.cache_dict = {}
   
   # New
   self.cache_config = CacheConfig(...)
   self.kv_cache = AdvancedKVCache(self.cache_config)
   ```

2. **Update cache access patterns:**
   ```python
   # Old
   if self.use_kv_cache and "key" in self.cache_dict:
       cached_key = self.cache_dict["key"]
   
   # New
   cached_kv = self.kv_cache.get_cache(layer_id, sequence_id, seq_len)
   if cached_kv:
       cached_key, cached_value = cached_kv
   ```

3. **Add performance monitoring:**
   ```python
   # Monitor cache performance
   stats = self.kv_cache.get_global_stats()
   if stats["global_hit_rate"] < 0.5:
       self.kv_cache.optimize_memory()
   ```

This advanced KV cache system provides production-ready memory management with significant performance improvements for your transformer models! 🎯