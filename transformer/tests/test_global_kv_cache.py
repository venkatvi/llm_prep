"""Unit tests for global KV cache implementation."""

import os
import sys
import time
import unittest
from unittest.mock import patch

import torch

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_kv_cache import CacheConfig, CacheEntry, GlobalKVCache


class TestCacheConfig(unittest.TestCase):
    """Test CacheConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        self.assertIsNone(config.max_cache_size)
        self.assertIsNone(config.max_memory_size)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CacheConfig(max_cache_size=10, max_memory_size=1.5)
        self.assertEqual(config.max_cache_size, 10)
        self.assertEqual(config.max_memory_size, 1.5)


class TestCacheEntry(unittest.TestCase):
    """Test CacheEntry dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.key = torch.rand(4, 8)
        self.value = torch.rand(4, 8)

    def test_entry_creation_with_timestamp(self):
        """Test creating entry with explicit timestamp."""
        timestamp = time.time()
        entry = CacheEntry(
            key=self.key,
            value=self.value,
            timestamp=timestamp,
            access_count=5,
            sequence_id="test_seq",
            layer_id=0,
        )
        self.assertEqual(entry.timestamp, timestamp)
        self.assertEqual(entry.access_count, 5)

    def test_entry_creation_without_timestamp(self):
        """Test creating entry with auto-generated timestamp."""
        with patch('time.time', return_value=123.456):
            entry = CacheEntry(
                key=self.key,
                value=self.value,
                timestamp=0.0,
                access_count=0,
                sequence_id="test_seq",
                layer_id=0,
            )
            self.assertEqual(entry.timestamp, 123.456)

    def test_bytes_calculation(self):
        """Test memory usage calculation."""
        entry = CacheEntry(
            key=self.key,
            value=self.value,
            timestamp=time.time(),
            access_count=0,
            sequence_id="test_seq",
            layer_id=0,
        )
        expected_bytes = (
            self.key.element_size() * self.key.nelement() +
            self.value.element_size() * self.value.nelement()
        )
        self.assertEqual(entry.bytes(), expected_bytes)


class TestGlobalKVCache(unittest.TestCase):
    """Test GlobalKVCache implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CacheConfig(max_cache_size=3, max_memory_size=0.001)
        self.cache = GlobalKVCache(config=self.config)
        
        # Create test entries
        self.entries = []
        for i in range(5):
            entry = CacheEntry(
                key=torch.rand(2, 4),
                value=torch.rand(2, 4),
                timestamp=time.time() + i,
                access_count=0,
                sequence_id=f"seq_{i % 3}",
                layer_id=i % 2,
            )
            self.entries.append(entry)

    def test_cache_initialization(self):
        """Test cache is properly initialized."""
        self.assertIsNotNone(self.cache.cache)
        self.assertEqual(len(self.cache.cache), 0)
        self.assertIsNotNone(self.cache.lock)

    def test_add_single_entry(self):
        """Test adding a single cache entry."""
        entry = self.entries[0]
        self.cache.add_entry(entry)
        
        # Check entry was added
        retrieved = self.cache.get_entry(entry.layer_id, entry.sequence_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0], entry)

    def test_add_multiple_entries_same_sequence(self):
        """Test adding multiple entries to same sequence."""
        entry1 = self.entries[0]
        entry2 = CacheEntry(
            key=torch.rand(3, 4),
            value=torch.rand(3, 4),
            timestamp=time.time(),
            access_count=0,
            sequence_id=entry1.sequence_id,
            layer_id=entry1.layer_id,
        )
        
        self.cache.add_entry(entry1)
        self.cache.add_entry(entry2)
        
        retrieved = self.cache.get_entry(entry1.layer_id, entry1.sequence_id)
        self.assertEqual(len(retrieved), 2)

    def test_add_entry_from_tensors(self):
        """Test adding entry directly from tensors."""
        key = torch.rand(4, 8)
        value = torch.rand(4, 8)
        
        self.cache.add_entry_from_tensors(
            key=key, value=value, sequence_id="test_seq", layer_id=0
        )
        
        retrieved = self.cache.get_entry(0, "test_seq")
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved), 1)
        
        # Check tensors are cloned (not same object)
        entry = retrieved[0]
        self.assertNotEqual(entry.key.data_ptr(), key.data_ptr())
        self.assertNotEqual(entry.value.data_ptr(), value.data_ptr())
        
        # But values should be equal
        self.assertTrue(torch.equal(entry.key, key))
        self.assertTrue(torch.equal(entry.value, value))

    def test_get_nonexistent_entry(self):
        """Test retrieving non-existent entry returns None."""
        result = self.cache.get_entry(999, "nonexistent")
        self.assertIsNone(result)

    def test_lru_access_tracking(self):
        """Test that LRU access tracking works correctly."""
        entry = self.entries[0]
        self.cache.add_entry(entry)
        
        # Access entry
        original_count = entry.access_count
        original_timestamp = entry.timestamp
        time.sleep(0.01)  # Small delay to ensure timestamp difference
        
        retrieved = self.cache.get_entry(entry.layer_id, entry.sequence_id)
        self.assertGreater(retrieved[0].access_count, original_count)
        self.assertGreater(retrieved[0].timestamp, original_timestamp)

    def test_eviction_per_layer(self):
        """Test evicting entries from specific layer."""
        # Add entries to layer 0
        for entry in self.entries[:3]:
            if entry.layer_id == 0:
                self.cache.add_entry(entry)
        
        # Evict from layer 0
        result = self.cache.evict_entry_per_layer(0)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 0)  # layer_id

    def test_eviction_empty_layer(self):
        """Test evicting from empty layer returns None."""
        result = self.cache.evict_entry_per_layer(999)
        self.assertIsNone(result)

    def test_global_eviction(self):
        """Test global eviction finds oldest entry across layers."""
        # Add entries with different timestamps
        entries_with_times = []
        for i, entry in enumerate(self.entries[:3]):
            entry.timestamp = time.time() - (10 - i)  # Older entries have smaller timestamps
            entries_with_times.append(entry)
            self.cache.add_entry(entry)
        
        # Global eviction should pick oldest
        result = self.cache.evict_entry()
        self.assertIsNotNone(result)

    def test_clear_layer(self):
        """Test clearing specific layer."""
        # Add entries to multiple layers
        for entry in self.entries:
            self.cache.add_entry(entry)
        
        # Clear layer 0
        self.cache.clear_layer(0)
        
        # Check layer 0 is empty but layer 1 still has entries
        stats = self.cache.stats()
        layer_0_entries = stats['layers'].get(0, {}).get('total_entries', 0)
        layer_1_entries = stats['layers'].get(1, {}).get('total_entries', 0)
        
        self.assertEqual(layer_0_entries, 0)
        self.assertGreater(layer_1_entries, 0)

    def test_clear_all(self):
        """Test clearing entire cache."""
        # Add entries
        for entry in self.entries:
            self.cache.add_entry(entry)
        
        # Clear all
        self.cache.clear()
        
        stats = self.cache.stats()
        self.assertEqual(stats['total_entries'], 0)
        self.assertEqual(stats['total_layers'], 0)

    def test_memory_calculation(self):
        """Test memory usage calculations."""
        entry = self.entries[0]
        self.cache.add_entry(entry)
        
        layer_bytes = self.cache.get_total_cache_size_per_layer_in_bytes(entry.layer_id)
        total_bytes = self.cache.get_total_cache_size_in_bytes()
        
        self.assertEqual(layer_bytes, entry.bytes())
        self.assertEqual(total_bytes, entry.bytes())

    def test_cache_size_limit_eviction(self):
        """Test eviction when cache size limit is exceeded."""
        # Create config with small size limit
        config = CacheConfig(max_cache_size=2)
        cache = GlobalKVCache(config=config)
        
        # Add more entries than limit
        for i in range(4):
            entry = CacheEntry(
                key=torch.rand(2, 4),
                value=torch.rand(2, 4),
                timestamp=time.time() + i,
                access_count=0,
                sequence_id=f"seq_{i}",
                layer_id=0,
            )
            cache.add_entry(entry)
        
        # Check that cache size doesn't exceed limit
        stats = cache.stats()
        layer_0_sequences = stats['layers'][0]['num_sequences']
        self.assertLessEqual(layer_0_sequences, 2)

    def test_stats_comprehensive(self):
        """Test comprehensive statistics reporting."""
        # Add diverse entries
        for entry in self.entries:
            self.cache.add_entry(entry)
        
        stats = self.cache.stats()
        
        # Check structure
        self.assertIn('total_layers', stats)
        self.assertIn('total_entries', stats)
        self.assertIn('total_size_bytes', stats)
        self.assertIn('layers', stats)
        
        # Check values
        self.assertGreater(stats['total_entries'], 0)
        self.assertGreater(stats['total_size_bytes'], 0)
        
        # Check layer-specific stats
        for layer_id, layer_stats in stats['layers'].items():
            self.assertIn('num_sequences', layer_stats)
            self.assertIn('total_entries', layer_stats)
            self.assertIn('size_bytes', layer_stats)

    def test_thread_safety_basic(self):
        """Basic thread safety test (lock acquisition)."""
        import threading
        
        results = []
        
        def add_entries():
            for i in range(10):
                entry = CacheEntry(
                    key=torch.rand(2, 4),
                    value=torch.rand(2, 4),
                    timestamp=time.time(),
                    access_count=0,
                    sequence_id=f"thread_seq_{i}",
                    layer_id=0,
                )
                self.cache.add_entry(entry)
                results.append(i)
        
        threads = [threading.Thread(target=add_entries) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should complete
        self.assertEqual(len(results), 30)  # 3 threads * 10 entries each


if __name__ == '__main__':
    unittest.main()