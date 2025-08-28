"""Global KV Cache implementation for transformer models."""

import time
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import torch


@dataclass
class CacheConfig:
    """Configuration for the global KV cache.

    Attributes:
        max_cache_size: Maximum number of sequences per layer (None = unlimited)
        max_memory_size: Maximum memory in MB per layer (None = unlimited)
    """

    max_cache_size: Optional[int] = None
    max_memory_size: Optional[float] = None


@dataclass
class CacheEntry:
    """Single entry in the KV cache.

    Attributes:
        key: Cached keys of shape [seq_len, embed_dim]
        value: Cached values of shape [seq_len, embed_dim]
        timestamp: Last access time for LRU eviction
        access_count: Number of times this entry has been accessed
        sequence_id: Identifier for the input sequence
        layer_id: Identifier for the transformer layer
    """

    key: torch.Tensor
    value: torch.Tensor
    timestamp: float
    access_count: int
    sequence_id: str
    layer_id: int

    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def bytes(self) -> int:
        """Calculate the memory size of this cache entry in bytes.

        Returns:
            Total memory usage in bytes for key and value tensors.
        """
        return (
            self.key.element_size() * self.key.nelement()
            + self.value.element_size() * self.value.nelement()
        )


@dataclass
class GlobalKVCache:
    """Global KV cache for transformer models.

    Thread-safe cache implementation with LRU eviction policy.
    Supports per-layer memory and size limits.

    Attributes:
        config: Configuration for the KV cache
        cache: Dictionary mapping layer_id to sequence caches
    """

    config: CacheConfig
    cache: Optional[Dict[int, OrderedDict[str, Deque[CacheEntry]]]] = None

    def __post_init__(self) -> None:
        """Initialize cache and thread lock."""
        self.cache = {}
        self.lock = threading.RLock()

    def _ensure_layer_cache(self, layer_id: int) -> None:
        """Ensure cache exists for given layer.

        Args:
            layer_id: ID of the transformer layer
        """
        if layer_id not in self.cache:
            self.cache[layer_id] = OrderedDict()

    def add_entry_from_tensors(self, key: torch.Tensor, value: torch.Tensor, sequence_id: str, layer_id: int) -> None:
        """Create and add a cache entry from key/value tensors.

        Args:
            key: Key tensor of shape [seq_len, embed_dim]
            value: Value tensor of shape [seq_len, embed_dim]
            sequence_id: Identifier for the input sequence
            layer_id: ID of the transformer layer
        """
        entry = CacheEntry(
            key=key.clone(),
            value=value.clone(),
            access_count=0,
            timestamp=0.0,
            sequence_id=sequence_id,
            layer_id=layer_id,
        )
        self.add_entry(entry)

    def add_entry(self, entry: CacheEntry) -> None:
        """Add a cache entry and maintain size/memory limits.

        Args:
            entry: Cache entry to add
        """
        with self.lock:
            layer_id = entry.layer_id
            sequence_id = entry.sequence_id

            self._ensure_layer_cache(layer_id)
            layer_cache = self.cache[layer_id]

            if sequence_id not in layer_cache:
                layer_cache[sequence_id] = deque()

            layer_cache[sequence_id].append(entry)
            entry.access_count += 1
            entry.timestamp = time.time()

            layer_cache.move_to_end(sequence_id)
            self._maybe_evict_cache_per_layer_no_lock(layer_id)

    def get_entry(self, layer_id: int, sequence_id: str) -> Optional[Deque[CacheEntry]]:
        """Retrieve cache entries for a specific layer and sequence.

        Args:
            layer_id: ID of the transformer layer
            sequence_id: Identifier for the sequence

        Returns:
            Deque of cache entries if found, None otherwise
        """
        with self.lock:
            layer_cache = self.cache.get(layer_id)
            if not layer_cache:
                return None

            entries = layer_cache.get(sequence_id)
            if entries and len(entries) > 0:
                # Update access statistics for LRU
                for entry in entries:
                    entry.access_count += 1
                    entry.timestamp = time.time()
                layer_cache.move_to_end(sequence_id)
                return entries
            return None

    def evict_entry_per_layer(
        self, layer_id: int
    ) -> Optional[Tuple[int, str, Optional[CacheEntry]]]:
        """Evict the least recently used entry from a specific layer.

        Args:
            layer_id: ID of the transformer layer

        Returns:
            Tuple of (layer_id, sequence_id, evicted_entry) or None
        """
        if not self.cache or layer_id not in self.cache or not self.cache[layer_id]:
            return None

        layer_cache = self.cache[layer_id]
        oldest_sequence_id = next(iter(layer_cache))
        entries = layer_cache[oldest_sequence_id]

        if entries:
            evicted_entry = entries.popleft()
            if not entries:
                del layer_cache[oldest_sequence_id]
            return (layer_id, oldest_sequence_id, evicted_entry)
        else:
            del layer_cache[oldest_sequence_id]
            return None

    def evict_entry(
        self, layer_id: Optional[int] = None
    ) -> Optional[Tuple[int, str, Optional[CacheEntry]]]:
        """Evict an entry from cache.

        Args:
            layer_id: Specific layer to evict from, or None for global eviction

        Returns:
            Tuple of (layer_id, sequence_id, evicted_entry) or None
        """
        with self.lock:
            if layer_id is not None:
                return self.evict_entry_per_layer(layer_id)

            if not self.cache:
                return None

            # Find layer with oldest entry
            oldest_layer_id = None
            oldest_timestamp = float("inf")

            for lid, layer_cache in self.cache.items():
                if not layer_cache:
                    continue
                first_sequence_id = next(iter(layer_cache))
                if not layer_cache[first_sequence_id]:
                    continue
                first_entry = layer_cache[first_sequence_id][0]
                if first_entry.timestamp < oldest_timestamp:
                    oldest_timestamp = first_entry.timestamp
                    oldest_layer_id = lid

            return (
                self.evict_entry_per_layer(oldest_layer_id)
                if oldest_layer_id is not None
                else None
            )

    def clear_layer(self, layer_id: int) -> None:
        """Clear all cache entries for a specific layer.

        Args:
            layer_id: ID of the transformer layer to clear
        """
        with self.lock:
            if self.cache and layer_id in self.cache:
                del self.cache[layer_id]

    def clear(self, layer_id: Optional[int] = None) -> None:
        """Clear cache entries.

        Args:
            layer_id: Specific layer to clear, or None to clear all
        """
        if layer_id is not None:
            self.clear_layer(layer_id)
        elif self.cache:
            with self.lock:
                self.cache.clear()

    def get_total_cache_size_per_layer_in_bytes(self, layer_id: int) -> int:
        """Calculate memory usage for a specific layer in bytes.

        Args:
            layer_id: ID of the transformer layer

        Returns:
            Total memory usage in bytes for the layer
        """
        if layer_id not in self.cache:
            return 0
        return sum(
            entry.bytes()
            for entries in self.cache[layer_id].values()
            for entry in entries
        )

    def get_total_cache_size_in_bytes(self) -> int:
        """Calculate total memory usage across all layers in bytes.

        Returns:
            Total memory usage in bytes across all layers
        """
        return sum(
            self.get_total_cache_size_per_layer_in_bytes(lid)
            for lid in self.cache.keys()
        )

    def _maybe_evict_cache_per_layer_no_lock(self, layer_id: int) -> None:
        """Evict entries from layer if limits exceeded (called with lock held).

        Args:
            layer_id: ID of the transformer layer
        """
        # Memory limit check
        if self.config.max_memory_size is not None:
            limit_in_bytes = self.config.max_memory_size * 1024 * 1024
            while (
                self.get_total_cache_size_per_layer_in_bytes(layer_id) > limit_in_bytes
            ):
                if not self.evict_entry_per_layer(layer_id):
                    break

        # Size limit check
        if self.config.max_cache_size is not None:
            while len(self.cache[layer_id]) > self.config.max_cache_size:
                if not self.evict_entry_per_layer(layer_id):
                    break

    def stats(self) -> Dict:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        with self.lock:
            total_entries = sum(
                sum(len(entries) for entries in layer_cache.values())
                for layer_cache in self.cache.values()
            )
            return {
                "total_layers": len(self.cache),
                "total_entries": total_entries,
                "total_size_bytes": self.get_total_cache_size_in_bytes(),
                "layers": {
                    layer_id: {
                        "num_sequences": len(layer_cache),
                        "total_entries": sum(
                            len(entries) for entries in layer_cache.values()
                        ),
                        "size_bytes": self.get_total_cache_size_per_layer_in_bytes(
                            layer_id
                        ),
                    }
                    for layer_id, layer_cache in self.cache.items()
                },
            }


if __name__ == "__main__":
    # Demo usage
    config = CacheConfig(max_cache_size=3, max_memory_size=0.001)
    cache = GlobalKVCache(config=config)

    for idx in range(10):
        entry = CacheEntry(
            key=torch.rand(32, 64),
            value=torch.rand(32, 64),
            access_count=0,
            timestamp=0.0,
            sequence_id=f"seq_{idx % 3}",
            layer_id=idx % 2,
        )
        cache.add_entry(entry)

    print("Final stats:", cache.stats())
