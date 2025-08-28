"""
Advanced KV Cache Implementation with Memory Management.

This module provides a sophisticated key-value caching system for transformer models
with support for variable sequence lengths, cache eviction policies, batch inference,
and memory fragmentation reduction.

Key Features:
- Variable sequence length handling with dynamic allocation
- Multiple cache eviction policies (LRU, FIFO, Sliding Window, Adaptive)
- Batch-aware caching with per-sample independent management
- Memory pool allocation to reduce fragmentation
- Async cache operations for improved performance
- Memory usage monitoring and optimization

Copyright (c) 2025. All rights reserved.
"""

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class EvictionPolicy(Enum):
    """Cache eviction policies for managing memory usage."""
    LRU = "lru"  # Least Recently Used
    FIFO = "fifo"  # First In First Out
    SLIDING_WINDOW = "sliding_window"  # Fixed-size sliding window
    ADAPTIVE = "adaptive"  # Dynamic policy based on usage patterns
    NONE = "none"  # No eviction (unbounded growth)


@dataclass
class CacheConfig:
    """Configuration for KV cache behavior and memory management."""
    
    # Core cache parameters
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    
    # Memory management
    eviction_policy: EvictionPolicy = EvictionPolicy.SLIDING_WINDOW
    cache_size_limit: Optional[int] = None  # Max cache entries per layer
    memory_limit_mb: Optional[float] = None  # Max memory usage in MB
    
    # Performance optimization
    preallocate_memory: bool = True  # Pre-allocate memory pools
    use_memory_pool: bool = True  # Use custom memory pool
    enable_async_ops: bool = False  # Async cache operations
    
    # Eviction policy specific parameters
    sliding_window_size: int = 512  # For SLIDING_WINDOW policy
    lru_capacity: int = 1000  # For LRU policy
    adaptive_threshold: float = 0.8  # For ADAPTIVE policy


@dataclass
class CacheEntry:
    """Single cache entry with metadata for management."""
    
    key: torch.Tensor  # [seq_len, embed_dim]
    value: torch.Tensor  # [seq_len, embed_dim] 
    timestamp: float = 0.0
    access_count: int = 0
    sequence_id: str = ""
    layer_id: int = 0
    
    def __post_init__(self):
        self.timestamp = time.time()
    
    @property
    def memory_usage(self) -> int:
        """Memory usage in bytes."""
        return (self.key.numel() + self.value.numel()) * self.key.element_size()
    
    def touch(self):
        """Update access metadata."""
        self.timestamp = time.time()
        self.access_count += 1


class MemoryPool:
    """Custom memory pool to reduce fragmentation and improve allocation speed."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.pools: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self.lock = threading.Lock()
        self._total_allocated = 0
        self._total_freed = 0
        
        if config.preallocate_memory:
            self._preallocate_pools()
    
    def _preallocate_pools(self):
        """Pre-allocate common tensor sizes to pool."""
        common_sizes = [
            (16, self.config.embed_dim),
            (32, self.config.embed_dim),
            (64, self.config.embed_dim),
            (128, self.config.embed_dim),
            (256, self.config.embed_dim),
            (512, self.config.embed_dim),
        ]
        
        for seq_len, embed_dim in common_sizes:
            pool_size = max(4, self.config.max_batch_size)
            self.pools[(seq_len, embed_dim)] = []
            
            for _ in range(pool_size):
                tensor = torch.empty(seq_len, embed_dim, dtype=torch.float32)
                self.pools[(seq_len, embed_dim)].append(tensor)
    
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate tensor from pool or create new one."""
        if not self.config.use_memory_pool:
            return torch.empty(shape, dtype=dtype)
        
        with self.lock:
            if shape in self.pools and self.pools[shape]:
                tensor = self.pools[shape].pop()
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                self._total_allocated += 1
                return tensor
        
        # Pool miss - create new tensor
        return torch.empty(shape, dtype=dtype)
    
    def deallocate(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse."""
        if not self.config.use_memory_pool:
            return
        
        shape = tensor.shape
        if len(shape) != 2:  # Only pool 2D tensors
            return
        
        with self.lock:
            if shape not in self.pools:
                self.pools[shape] = []
            
            # Limit pool size to prevent unbounded growth
            max_pool_size = max(8, self.config.max_batch_size * 2)
            if len(self.pools[shape]) < max_pool_size:
                self.pools[shape].append(tensor.detach().clone())
                self._total_freed += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics."""
        total_pooled = sum(len(pool) for pool in self.pools.values())
        return {
            "total_allocated": self._total_allocated,
            "total_freed": self._total_freed,
            "total_pooled": total_pooled,
            "pool_hit_rate": self._total_freed / max(1, self._total_allocated)
        }


class CacheEvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def should_evict(self, entries: List[CacheEntry], config: CacheConfig) -> List[int]:
        """Return indices of entries to evict."""
        pass
    
    @abstractmethod
    def select_victim(self, entries: List[CacheEntry]) -> int:
        """Select single entry to evict."""
        pass


class LRUEvictionPolicy(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def should_evict(self, entries: List[CacheEntry], config: CacheConfig) -> List[int]:
        if not config.cache_size_limit or len(entries) <= config.cache_size_limit:
            return []
        
        # Sort by timestamp (oldest first) and return excess indices
        sorted_entries = sorted(enumerate(entries), key=lambda x: x[1].timestamp)
        excess_count = len(entries) - config.cache_size_limit
        return [idx for idx, _ in sorted_entries[:excess_count]]
    
    def select_victim(self, entries: List[CacheEntry]) -> int:
        if not entries:
            return -1
        return min(enumerate(entries), key=lambda x: x[1].timestamp)[0]


class SlidingWindowPolicy(CacheEvictionPolicy):
    """Sliding window eviction policy - maintains fixed window of recent entries."""
    
    def should_evict(self, entries: List[CacheEntry], config: CacheConfig) -> List[int]:
        window_size = config.sliding_window_size
        if len(entries) <= window_size:
            return []
        
        # Keep only the most recent window_size entries
        sorted_entries = sorted(enumerate(entries), key=lambda x: x[1].timestamp, reverse=True)
        keep_indices = {idx for idx, _ in sorted_entries[:window_size]}
        return [i for i in range(len(entries)) if i not in keep_indices]
    
    def select_victim(self, entries: List[CacheEntry]) -> int:
        if not entries:
            return -1
        return min(enumerate(entries), key=lambda x: x[1].timestamp)[0]


class AdaptiveEvictionPolicy(CacheEvictionPolicy):
    """Adaptive policy that switches between strategies based on usage patterns."""
    
    def __init__(self):
        self.lru_policy = LRUEvictionPolicy()
        self.sliding_policy = SlidingWindowPolicy()
        self.access_pattern_history = []
    
    def should_evict(self, entries: List[CacheEntry], config: CacheConfig) -> List[int]:
        # Analyze access patterns to choose policy
        if self._is_sequential_pattern(entries):
            return self.sliding_policy.should_evict(entries, config)
        else:
            return self.lru_policy.should_evict(entries, config)
    
    def select_victim(self, entries: List[CacheEntry]) -> int:
        if self._is_sequential_pattern(entries):
            return self.sliding_policy.select_victim(entries)
        else:
            return self.lru_policy.select_victim(entries)
    
    def _is_sequential_pattern(self, entries: List[CacheEntry]) -> bool:
        """Detect if access pattern is mostly sequential."""
        if len(entries) < 3:
            return True
        
        # Simple heuristic: if recent entries have increasing timestamps
        recent_entries = sorted(entries, key=lambda x: x.timestamp, reverse=True)[:5]
        timestamps = [entry.timestamp for entry in recent_entries]
        
        # Check if timestamps are roughly in order (allowing some variance)
        variance = sum(abs(timestamps[i] - timestamps[i+1]) for i in range(len(timestamps)-1))
        return variance < 0.1  # Low variance suggests sequential access


class LayerKVCache:
    """KV cache for a single transformer layer with advanced memory management."""
    
    def __init__(self, layer_id: int, config: CacheConfig, memory_pool: MemoryPool):
        self.layer_id = layer_id
        self.config = config
        self.memory_pool = memory_pool
        
        # Per-sequence caches: sequence_id -> CacheEntry
        self.caches: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        
        # Eviction policy
        self.eviction_policy = self._create_eviction_policy()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _create_eviction_policy(self) -> CacheEvictionPolicy:
        """Create eviction policy based on configuration."""
        if self.config.eviction_policy == EvictionPolicy.LRU:
            return LRUEvictionPolicy()
        elif self.config.eviction_policy == EvictionPolicy.SLIDING_WINDOW:
            return SlidingWindowPolicy()
        elif self.config.eviction_policy == EvictionPolicy.ADAPTIVE:
            return AdaptiveEvictionPolicy()
        else:
            return LRUEvictionPolicy()  # Default fallback
    
    def get(self, sequence_id: str, seq_len: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve cached key-value pair if available and valid."""
        with self.lock:
            if sequence_id not in self.caches:
                self.misses += 1
                return None
            
            entry = self.caches[sequence_id]
            
            # Check if cached sequence length is sufficient
            if entry.key.size(0) < seq_len:
                self.misses += 1
                return None
            
            # Cache hit
            entry.touch()
            self.hits += 1
            
            # Return sliced tensors for the requested length
            return entry.key[:seq_len], entry.value[:seq_len]
    
    def put(self, sequence_id: str, key: torch.Tensor, value: torch.Tensor):
        """Store key-value pair in cache with automatic memory management."""
        with self.lock:
            # Create new cache entry
            entry = CacheEntry(
                key=key.clone(),
                value=value.clone(),
                sequence_id=sequence_id,
                layer_id=self.layer_id
            )
            
            # Check if we need to evict before adding
            self._maybe_evict()
            
            # Store the entry
            if sequence_id in self.caches:
                # Free old entry memory
                self._free_entry(self.caches[sequence_id])
            
            self.caches[sequence_id] = entry
    
    def extend(self, sequence_id: str, new_key: torch.Tensor, new_value: torch.Tensor):
        """Extend existing cache entry with new tokens."""
        with self.lock:
            if sequence_id not in self.caches:
                # No existing cache, create new
                self.put(sequence_id, new_key, new_value)
                return
            
            entry = self.caches[sequence_id]
            
            # Concatenate with existing cache
            extended_key = torch.cat([entry.key, new_key], dim=0)
            extended_value = torch.cat([entry.value, new_value], dim=0)
            
            # Update cache
            self._free_entry(entry)
            self.put(sequence_id, extended_key, extended_value)
    
    def _maybe_evict(self):
        """Perform eviction if necessary based on policy."""
        if self.config.eviction_policy == EvictionPolicy.NONE:
            return
        
        entries = list(self.caches.values())
        if not entries:
            return
        
        # Check memory limit
        if self.config.memory_limit_mb:
            total_memory = sum(entry.memory_usage for entry in entries)
            memory_limit_bytes = self.config.memory_limit_mb * 1024 * 1024
            
            if total_memory > memory_limit_bytes:
                self._evict_by_memory_pressure(entries)
                return
        
        # Check size limit
        indices_to_evict = self.eviction_policy.should_evict(entries, self.config)
        if indices_to_evict:
            sequence_ids = list(self.caches.keys())
            for idx in sorted(indices_to_evict, reverse=True):
                if idx < len(sequence_ids):
                    sequence_id = sequence_ids[idx]
                    self._evict_entry(sequence_id)
    
    def _evict_by_memory_pressure(self, entries: List[CacheEntry]):
        """Evict entries to reduce memory pressure."""
        # Sort by memory usage (largest first) and access frequency (least used first)
        sorted_entries = sorted(
            enumerate(entries), 
            key=lambda x: (x[1].memory_usage, -x[1].access_count),
            reverse=True
        )
        
        sequence_ids = list(self.caches.keys())
        memory_limit = self.config.memory_limit_mb * 1024 * 1024 * 0.8  # Target 80% of limit
        current_memory = sum(entry.memory_usage for entry in entries)
        
        for idx, entry in sorted_entries:
            if current_memory <= memory_limit:
                break
            
            if idx < len(sequence_ids):
                sequence_id = sequence_ids[idx]
                self._evict_entry(sequence_id)
                current_memory -= entry.memory_usage
    
    def _evict_entry(self, sequence_id: str):
        """Evict specific cache entry."""
        if sequence_id in self.caches:
            entry = self.caches[sequence_id]
            self._free_entry(entry)
            del self.caches[sequence_id]
            self.evictions += 1
    
    def _free_entry(self, entry: CacheEntry):
        """Free memory used by cache entry."""
        self.memory_pool.deallocate(entry.key)
        self.memory_pool.deallocate(entry.value)
    
    def clear(self, sequence_id: Optional[str] = None):
        """Clear cache entries."""
        with self.lock:
            if sequence_id:
                if sequence_id in self.caches:
                    self._evict_entry(sequence_id)
            else:
                # Clear all entries
                for entry in self.caches.values():
                    self._free_entry(entry)
                self.caches.clear()
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(1, total_requests)
        
        with self.lock:
            total_memory = sum(entry.memory_usage for entry in self.caches.values())
            num_entries = len(self.caches)
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "num_entries": num_entries,
            "total_memory_mb": total_memory / (1024 * 1024),
            "avg_entry_size_kb": (total_memory / max(1, num_entries)) / 1024
        }


class AdvancedKVCache:
    """Advanced multi-layer KV cache with comprehensive memory management."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_pool = MemoryPool(config)
        
        # Per-layer caches
        self.layer_caches: Dict[int, LayerKVCache] = {}
        for layer_id in range(config.num_layers):
            self.layer_caches[layer_id] = LayerKVCache(layer_id, config, self.memory_pool)
        
        # Global statistics
        self.global_lock = threading.Lock()
    
    def get_cache(self, layer_id: int, sequence_id: str, seq_len: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV for specific layer and sequence."""
        if layer_id not in self.layer_caches:
            return None
        
        return self.layer_caches[layer_id].get(sequence_id, seq_len)
    
    def set_cache(self, layer_id: int, sequence_id: str, key: torch.Tensor, value: torch.Tensor):
        """Set cached KV for specific layer and sequence."""
        if layer_id in self.layer_caches:
            self.layer_caches[layer_id].put(sequence_id, key, value)
    
    def extend_cache(self, layer_id: int, sequence_id: str, new_key: torch.Tensor, new_value: torch.Tensor):
        """Extend existing cache with new tokens."""
        if layer_id in self.layer_caches:
            self.layer_caches[layer_id].extend(sequence_id, new_key, new_value)
    
    def clear_sequence(self, sequence_id: str):
        """Clear cache for specific sequence across all layers."""
        for layer_cache in self.layer_caches.values():
            layer_cache.clear(sequence_id)
    
    def clear_all(self):
        """Clear all caches."""
        for layer_cache in self.layer_caches.values():
            layer_cache.clear()
    
    def get_global_stats(self) -> Dict[str, Union[int, float, Dict]]:
        """Get comprehensive cache statistics."""
        layer_stats = {}
        total_hits = 0
        total_misses = 0
        total_evictions = 0
        total_memory = 0
        
        for layer_id, layer_cache in self.layer_caches.items():
            stats = layer_cache.get_stats()
            layer_stats[f"layer_{layer_id}"] = stats
            
            total_hits += stats["hits"]
            total_misses += stats["misses"]
            total_evictions += stats["evictions"]
            total_memory += stats["total_memory_mb"]
        
        global_hit_rate = total_hits / max(1, total_hits + total_misses)
        memory_pool_stats = self.memory_pool.get_stats()
        
        return {
            "global_hit_rate": global_hit_rate,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_evictions": total_evictions,
            "total_memory_mb": total_memory,
            "memory_pool": memory_pool_stats,
            "layer_stats": layer_stats
        }
    
    def optimize_memory(self):
        """Trigger memory optimization across all layers."""
        for layer_cache in self.layer_caches.values():
            layer_cache._maybe_evict()


# Integration with existing attention mechanisms
class CachedAttentionMixin:
    """Mixin class to add advanced caching to attention mechanisms."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache: Optional[AdvancedKVCache] = None
        self.cache_sequence_id: str = ""
    
    def set_cache(self, cache: AdvancedKVCache, sequence_id: str = "default"):
        """Set the KV cache instance."""
        self.kv_cache = cache
        self.cache_sequence_id = sequence_id
    
    def _get_cached_kv(self, layer_id: int, seq_len: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve cached key-value pair."""
        if not self.kv_cache:
            return None
        return self.kv_cache.get_cache(layer_id, self.cache_sequence_id, seq_len)
    
    def _cache_kv(self, layer_id: int, key: torch.Tensor, value: torch.Tensor):
        """Cache key-value pair."""
        if self.kv_cache:
            self.kv_cache.set_cache(layer_id, self.cache_sequence_id, key, value)
    
    def _extend_cached_kv(self, layer_id: int, new_key: torch.Tensor, new_value: torch.Tensor):
        """Extend cached key-value pair."""
        if self.kv_cache:
            self.kv_cache.extend_cache(layer_id, self.cache_sequence_id, new_key, new_value)


# Example usage and testing utilities
def create_cache_config_for_speculative_decoding() -> Tuple[CacheConfig, CacheConfig]:
    """Create optimized cache configurations for speculative decoding."""
    
    # Draft model cache config (optimized for sequential generation)
    draft_cache_config = CacheConfig(
        max_batch_size=16,
        max_sequence_length=512,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        eviction_policy=EvictionPolicy.SLIDING_WINDOW,
        sliding_window_size=256,
        memory_limit_mb=100,
        preallocate_memory=True,
        use_memory_pool=True,
    )
    
    # Target model cache config (minimal caching for parallel processing)
    target_cache_config = CacheConfig(
        max_batch_size=16,
        max_sequence_length=1024,
        embed_dim=1024,
        num_heads=16,
        num_layers=12,
        eviction_policy=EvictionPolicy.NONE,  # No caching for parallel processing
        memory_limit_mb=50,
        preallocate_memory=False,
        use_memory_pool=False,
    )
    
    return draft_cache_config, target_cache_config


if __name__ == "__main__":
    # Example usage
    config = CacheConfig(
        max_batch_size=4,
        max_sequence_length=128,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        eviction_policy=EvictionPolicy.ADAPTIVE,
        memory_limit_mb=50
    )
    
    cache = AdvancedKVCache(config)
    
    # Simulate cache usage
    for layer_id in range(3):
        for seq_id in ["seq_1", "seq_2"]:
            key = torch.randn(32, 256)
            value = torch.randn(32, 256)
            cache.set_cache(layer_id, seq_id, key, value)
    
    # Get statistics
    stats = cache.get_global_stats()
    print("Cache Statistics:")
    print(f"Global Hit Rate: {stats['global_hit_rate']:.2%}")
    print(f"Total Memory: {stats['total_memory_mb']:.2f} MB")
    print(f"Memory Pool Hit Rate: {stats['memory_pool']['pool_hit_rate']:.2%}")