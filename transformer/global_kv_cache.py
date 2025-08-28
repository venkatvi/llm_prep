import torch 
from dataclasses import dataclass 
from typing import Optional, Tuple
from collections import OrderedDict
import time 

@dataclass 
class CacheConfig: 
    # # Size of the model 
    # embed_dim: int 
    # num_heads: int 
    # num_groups: int
    # max_seq_len: int 
    # max_batch_size: int

    # # Eviction policy 
    # eviction_policy: str = "lru"

    # Limits 
    max_cache_size: Optional[int] = None  # Max entries per layer 
    max_memory_size: Optional[float] = None  # Max memory in MB per layer 

@dataclass
class CacheEntry:
    """Single entry in the KV cache.

    Attributes:
        key (torch.Tensor): Cached keys of shape [seq_len, embed_dim]
        value (torch.Tensor): Cached values of shape [seq_len, embed_dim]
        timestamp (float): Last access time for LRU eviction
        access_count (int): Number of times this entry has been accessed
        sequence_id (str): Identifier for the input sequence
        layer_id (int): Identifier for the transformer layer
    """ 
    key: torch.Tensor  # [seq_len, embed_dim]
    value: torch.Tensor  # [seq_len, embed_dim]
    timestamp: float
    access_count: int
    sequence_id: str
    layer_id: int   

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def bytes(self) -> int:
        """Calculate the memory size of this cache entry in bytes."""
        return self.key.element_size() * self.key.nelement() + \
               self.value.element_size() * self.value.nelement()
@dataclass
class GlobalKVCache:
    """Global KV cache for transformer models.

    Attributes:
        config (CacheConfig): Configuration for the KV cache
        cache (dict): Dictionary mapping (layer_id, sequence_id) to CacheEntry
    """
    config: CacheConfig
    cache: Optional[OrderedDict[Tuple[int, str], CacheEntry]] = None

    def __post_init__(self):
        self.cache = OrderedDict()
        
    
    def add_entry(self, entry: CacheEntry): 
        cache_key = (entry.layer_id, entry.sequence_id)
        if cache_key in self.cache:
            raise ValueError(f"Entry for layer {entry.layer_id} and sequence {entry.sequence_id} already exists in cache.")
        self.cache[cache_key] = entry
        self.cache.move_to_end(cache_key) # most recently used 
        self._maybe_evict_cache()
    
    def get_entry(self, layer_id: int, sequence_id: str) -> Optional[CacheEntry]: 
        cache_key = (layer_id, sequence_id)
        entry = self.cache.get(cache_key, None)
        if entry:
            entry.access_count += 1
            # Update timestamp for LRU
            import time
            entry.timestamp = time.time()
            self.cache.move_to_end(cache_key) # most recently used
        else: 
            print(f"Cache miss for layer {layer_id} and sequence {sequence_id}")
        return entry
    
    def evict_entry(self): 
        if self.cache is None or len(self.cache) == 0: 
            print("Cache is empty, nothing to evict")
            return None
        return self.cache.popitem(last=False)   # remove least recently used
        
    def clear(self): 
        self.cache.clear()

    def get_total_cache_size_in_bytes(self) -> int:
        return sum(e.bytes() for e in self.cache.values())
    
    def _maybe_evict_cache(self): 
        if self.config.max_memory_size: 
            limit_in_bytes = self.config.max_memory_size * 1024 * 1024
            while self.get_total_cache_size_in_bytes() > limit_in_bytes:
                print("Evicting entry to maintain memory limit")
                self.evict_entry()
            
        if self.config.max_cache_size: 
            while len(self.cache) > self.config.max_cache_size: 
                print("Evicting entry to maintain cache size limit")
                self.evict_entry()

    def stats(self): 
        total_entries = len(self.cache)
        total_size_bytes = self.get_total_cache_size_in_bytes()
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size_bytes
        }

if __name__ == "__main__": 
    config = CacheConfig(max_cache_size=10, max_memory_size=0.02) # MB 
    gKVCache = GlobalKVCache(config=config)
    kv_bs = [10, 20 ,30, 10, 20, 30, 1, 2, 3, 1, 2, 3, 100, 200, 300]
    kv_sl = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4]
    for idx in range(25): 
       entry = CacheEntry(
           key=torch.rand(kv_bs[idx%15], kv_sl[idx%15]), 
           value=torch.rand(kv_bs[idx%15], kv_sl[idx%15]),
           access_count=0,
           timestamp=0.0,
           sequence_id=f"seq{idx%5}",
           layer_id=idx%3
       )
       gKVCache.add_entry(entry)
       #gKVCache.add_entry(entry)
       print(f"Added entry {idx}, cache stats: {gKVCache.stats()}")