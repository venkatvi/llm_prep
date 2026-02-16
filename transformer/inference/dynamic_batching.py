"""
Copyright (c) 2025. All rights reserved.
"""

"""
Advanced dynamic batching for variable sequence lengths and shapes.

This module implements sophisticated batching strategies that handle:
- Variable sequence lengths efficiently
- Dynamic batch formation based on memory constraints
- Optimal padding and bucketing strategies
- Continuous batching for streaming inference
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Iterator
import heapq
import threading
import time
import math
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import numpy as np


class BatchingStrategy(Enum):
    """Different batching strategies."""
    NAIVE = "naive"  # Simple padding to max length
    BUCKETED = "bucketed"  # Group similar lengths
    MEMORY_AWARE = "memory_aware"  # Consider memory constraints
    CONTINUOUS = "continuous"  # Continuous batching for streaming


@dataclass
class BatchingConfig:
    """Configuration for dynamic batching."""
    strategy: BatchingStrategy = BatchingStrategy.MEMORY_AWARE
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    bucket_size: int = 128  # Length difference for bucketing
    memory_limit_mb: int = 4096  # GPU memory limit
    padding_token_id: int = 0
    batch_timeout_ms: float = 50.0
    enable_continuous_batching: bool = True
    prefill_chunk_size: int = 512
    max_total_tokens: int = 32768  # Total tokens across batch


class MemoryEstimator:
    """Estimates memory usage for different batch configurations."""
    
    def __init__(self, model_hidden_size: int, num_layers: int, num_heads: int):
        self.hidden_size = model_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = model_hidden_size // num_heads
        
    def estimate_batch_memory(
        self,
        batch_size: int,
        sequence_length: int,
        dtype: torch.dtype = torch.float16
    ) -> int:
        """Estimate memory usage in bytes for a batch."""
        # Element size in bytes
        if dtype == torch.float16:
            element_size = 2
        elif dtype == torch.float32:
            element_size = 4
        elif dtype == torch.bfloat16:
            element_size = 2
        else:
            element_size = 4
            
        # Activation memory
        activation_memory = (
            batch_size * sequence_length * self.hidden_size * element_size * 
            (4 + self.num_layers * 2)  # input, output, intermediate activations per layer
        )
        
        # KV cache memory
        kv_cache_memory = (
            batch_size * sequence_length * self.hidden_size * element_size * 
            self.num_layers * 2  # key and value for each layer
        )
        
        # Attention scores memory (temporary)
        attention_memory = (
            batch_size * self.num_heads * sequence_length * sequence_length * 
            element_size
        )
        
        # Total estimated memory
        total_memory = activation_memory + kv_cache_memory + attention_memory
        
        return int(total_memory * 1.2)  # Add 20% safety margin
    
    def find_optimal_batch_size(
        self,
        sequence_lengths: List[int],
        memory_limit: int,
        max_batch_size: int
    ) -> int:
        """Find optimal batch size given memory constraints."""
        if not sequence_lengths:
            return 0
            
        max_seq_len = max(sequence_lengths)
        
        # Binary search for optimal batch size
        left, right = 1, min(max_batch_size, len(sequence_lengths))
        optimal_size = 1
        
        while left <= right:
            mid = (left + right) // 2
            estimated_memory = self.estimate_batch_memory(mid, max_seq_len)
            
            if estimated_memory <= memory_limit:
                optimal_size = mid
                left = mid + 1
            else:
                right = mid - 1
                
        return optimal_size


class SequenceBucketer:
    """Groups sequences into buckets by length for efficient batching."""
    
    def __init__(self, bucket_size: int = 128):
        self.bucket_size = bucket_size
        self.buckets: Dict[int, List] = defaultdict(list)
        
    def add_sequence(self, item: Any, length: int):
        """Add sequence to appropriate bucket."""
        bucket_id = (length + self.bucket_size - 1) // self.bucket_size
        self.buckets[bucket_id].append((item, length))
        
    def get_batch_from_bucket(self, max_batch_size: int) -> Tuple[List, int]:
        """Get a batch from the fullest bucket."""
        if not self.buckets:
            return [], 0
            
        # Find bucket with most items
        best_bucket_id = max(self.buckets.keys(), key=lambda x: len(self.buckets[x]))
        bucket = self.buckets[best_bucket_id]
        
        # Extract batch
        batch_size = min(max_batch_size, len(bucket))
        batch_items = [bucket.pop(0) for _ in range(batch_size)]
        
        # Remove empty bucket
        if not bucket:
            del self.buckets[best_bucket_id]
            
        return batch_items, best_bucket_id * self.bucket_size
    
    def get_all_items(self) -> List[Tuple[Any, int]]:
        """Get all items from all buckets."""
        items = []
        for bucket in self.buckets.values():
            items.extend(bucket)
        return items
    
    def clear(self):
        """Clear all buckets."""
        self.buckets.clear()


class ContinuousBatcher:
    """Implements continuous batching for streaming inference."""
    
    def __init__(self, config: BatchingConfig, memory_estimator: MemoryEstimator):
        self.config = config
        self.memory_estimator = memory_estimator
        self.active_requests = []  # Currently processing requests
        self.pending_requests = deque()  # Waiting to be added
        self.lock = threading.Lock()
        
    def add_request(self, request: Any, sequence_length: int):
        """Add new request to pending queue."""
        with self.lock:
            self.pending_requests.append((request, sequence_length))
            
    def update_batch(self) -> Tuple[List, List]:
        """Update current batch with new requests and completed ones."""
        with self.lock:
            # Remove completed requests
            active_requests = [
                (req, length) for req, length in self.active_requests
                if not self._is_request_complete(req)
            ]
            
            # Calculate current memory usage
            current_memory = sum(
                self.memory_estimator.estimate_batch_memory(1, length)
                for _, length in active_requests
            )
            
            # Add new requests if memory allows
            new_requests = []
            while (self.pending_requests and 
                   len(active_requests) + len(new_requests) < self.config.max_batch_size):
                
                req, length = self.pending_requests[0]
                additional_memory = self.memory_estimator.estimate_batch_memory(1, length)
                
                if current_memory + additional_memory <= self.config.memory_limit_mb * 1024 * 1024:
                    new_requests.append(self.pending_requests.popleft())
                    current_memory += additional_memory
                else:
                    break
                    
            # Update active requests
            self.active_requests = active_requests + new_requests
            
            # Separate requests by type
            continuing_requests = [(req, length) for req, length in active_requests]
            new_batch_requests = new_requests
            
            return continuing_requests, new_batch_requests
    
    def _is_request_complete(self, request: Any) -> bool:
        """Check if request is complete (to be implemented based on request type)."""
        # This would check if the request has finished generation
        return hasattr(request, 'is_complete') and request.is_complete


class DynamicBatcher:
    """Advanced dynamic batcher with multiple strategies."""
    
    def __init__(self, config: BatchingConfig, memory_estimator: Optional[MemoryEstimator] = None):
        self.config = config
        self.memory_estimator = memory_estimator
        self.bucketer = SequenceBucketer(config.bucket_size)
        self.continuous_batcher = None
        
        if config.enable_continuous_batching and memory_estimator:
            self.continuous_batcher = ContinuousBatcher(config, memory_estimator)
            
        self.pending_items = []
        self.lock = threading.Lock()
        
    def add_items(self, items: List[Tuple[Any, int]]):
        """Add items with their sequence lengths."""
        with self.lock:
            if self.config.strategy == BatchingStrategy.CONTINUOUS and self.continuous_batcher:
                for item, length in items:
                    self.continuous_batcher.add_request(item, length)
            else:
                self.pending_items.extend(items)
                
    def get_next_batch(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Get next optimized batch."""
        if self.config.strategy == BatchingStrategy.CONTINUOUS and self.continuous_batcher:
            return self._get_continuous_batch()
        else:
            return self._get_static_batch()
            
    def _get_static_batch(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Get batch using static batching strategies."""
        with self.lock:
            if not self.pending_items:
                return [], {}
                
            if self.config.strategy == BatchingStrategy.NAIVE:
                return self._get_naive_batch()
            elif self.config.strategy == BatchingStrategy.BUCKETED:
                return self._get_bucketed_batch()
            elif self.config.strategy == BatchingStrategy.MEMORY_AWARE:
                return self._get_memory_aware_batch()
            else:
                return self._get_naive_batch()
                
    def _get_naive_batch(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Simple batching - take first N items."""
        batch_size = min(self.config.max_batch_size, len(self.pending_items))
        batch_items = self.pending_items[:batch_size]
        self.pending_items = self.pending_items[batch_size:]
        
        items = [item for item, _ in batch_items]
        lengths = [length for _, length in batch_items]
        max_length = min(max(lengths), self.config.max_sequence_length)
        
        return items, {
            'strategy': 'naive',
            'batch_size': len(items),
            'max_length': max_length,
            'lengths': lengths
        }
        
    def _get_bucketed_batch(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Bucketed batching - group similar lengths."""
        # Add items to buckets
        for item, length in self.pending_items:
            self.bucketer.add_sequence(item, length)
        self.pending_items.clear()
        
        # Get batch from fullest bucket
        batch_items, bucket_length = self.bucketer.get_batch_from_bucket(
            self.config.max_batch_size
        )
        
        if not batch_items:
            return [], {}
            
        items = [item for item, _ in batch_items]
        lengths = [length for _, length in batch_items]
        max_length = min(max(lengths), self.config.max_sequence_length)
        
        return items, {
            'strategy': 'bucketed',
            'batch_size': len(items),
            'max_length': max_length,
            'lengths': lengths,
            'bucket_length': bucket_length
        }
        
    def _get_memory_aware_batch(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Memory-aware batching considering GPU memory constraints."""
        if not self.memory_estimator:
            return self._get_bucketed_batch()
            
        # Sort by sequence length for better packing
        self.pending_items.sort(key=lambda x: x[1])
        
        # Find optimal batch considering memory
        batch_items = []
        current_memory = 0
        memory_limit = self.config.memory_limit_mb * 1024 * 1024
        
        for item, length in self.pending_items:
            if len(batch_items) >= self.config.max_batch_size:
                break
                
            # Estimate memory for this batch + new item
            test_lengths = [l for _, l in batch_items] + [length]
            max_test_length = max(test_lengths)
            estimated_memory = self.memory_estimator.estimate_batch_memory(
                len(test_lengths), max_test_length
            )
            
            if estimated_memory <= memory_limit:
                batch_items.append((item, length))
                current_memory = estimated_memory
            else:
                break
                
        # Remove selected items from pending
        for item in batch_items:
            self.pending_items.remove(item)
            
        if not batch_items:
            return [], {}
            
        items = [item for item, _ in batch_items]
        lengths = [length for _, length in batch_items]
        max_length = min(max(lengths), self.config.max_sequence_length)
        
        return items, {
            'strategy': 'memory_aware',
            'batch_size': len(items),
            'max_length': max_length,
            'lengths': lengths,
            'estimated_memory_mb': current_memory // (1024 * 1024),
            'memory_utilization': current_memory / memory_limit
        }
        
    def _get_continuous_batch(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Get batch for continuous batching."""
        continuing, new = self.continuous_batcher.update_batch()
        
        all_items = continuing + new
        if not all_items:
            return [], {}
            
        items = [item for item, _ in all_items]
        lengths = [length for _, length in all_items]
        max_length = min(max(lengths), self.config.max_sequence_length) if lengths else 0
        
        return items, {
            'strategy': 'continuous',
            'batch_size': len(items),
            'max_length': max_length,
            'lengths': lengths,
            'continuing_requests': len(continuing),
            'new_requests': len(new)
        }


def create_padded_batch(
    input_ids_list: List[torch.Tensor],
    attention_masks_list: Optional[List[torch.Tensor]] = None,
    padding_token_id: int = 0,
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """Create padded batch from list of input sequences."""
    if not input_ids_list:
        return {}
        
    batch_size = len(input_ids_list)
    
    # Determine max length
    if max_length is None:
        max_length = max(ids.size(-1) for ids in input_ids_list)
        
    # Get device and dtype from first tensor
    if device is None:
        device = input_ids_list[0].device
    dtype = input_ids_list[0].dtype
    
    # Create padded tensors
    padded_input_ids = torch.full(
        (batch_size, max_length),
        padding_token_id,
        dtype=dtype,
        device=device
    )
    
    padded_attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
        device=device
    )
    
    # Fill padded tensors
    for i, input_ids in enumerate(input_ids_list):
        seq_len = min(input_ids.size(-1), max_length)
        padded_input_ids[i, :seq_len] = input_ids[:seq_len]
        padded_attention_mask[i, :seq_len] = 1
        
        # Use provided attention mask if available
        if attention_masks_list and attention_masks_list[i] is not None:
            padded_attention_mask[i, :seq_len] = attention_masks_list[i][:seq_len]
            
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask
    }


def calculate_padding_efficiency(sequence_lengths: List[int]) -> Dict[str, float]:
    """Calculate padding efficiency metrics for a batch."""
    if not sequence_lengths:
        return {}
        
    max_length = max(sequence_lengths)
    total_tokens = sum(sequence_lengths)
    total_padded_tokens = len(sequence_lengths) * max_length
    padding_tokens = total_padded_tokens - total_tokens
    
    return {
        'total_tokens': total_tokens,
        'padding_tokens': padding_tokens,
        'efficiency': total_tokens / total_padded_tokens,
        'waste_ratio': padding_tokens / total_padded_tokens,
        'max_length': max_length,
        'min_length': min(sequence_lengths),
        'avg_length': total_tokens / len(sequence_lengths),
        'length_variance': np.var(sequence_lengths)
    }


class BatchingOptimizer:
    """Optimizes batching strategies based on historical data."""
    
    def __init__(self):
        self.batch_history = deque(maxlen=1000)
        self.strategy_performance = defaultdict(list)
        
    def record_batch(self, batch_info: Dict[str, Any], processing_time: float, memory_usage: int):
        """Record batch performance."""
        record = {
            'timestamp': time.time(),
            'strategy': batch_info.get('strategy', 'unknown'),
            'batch_size': batch_info.get('batch_size', 0),
            'max_length': batch_info.get('max_length', 0),
            'processing_time': processing_time,
            'memory_usage': memory_usage,
            'efficiency': batch_info.get('efficiency', 0)
        }
        
        self.batch_history.append(record)
        self.strategy_performance[record['strategy']].append({
            'processing_time': processing_time,
            'memory_usage': memory_usage,
            'efficiency': record['efficiency']
        })
        
    def get_best_strategy(self) -> str:
        """Get best performing strategy based on historical data."""
        if not self.strategy_performance:
            return BatchingStrategy.MEMORY_AWARE.value
            
        strategy_scores = {}
        
        for strategy, records in self.strategy_performance.items():
            if not records:
                continue
                
            avg_time = np.mean([r['processing_time'] for r in records])
            avg_memory = np.mean([r['memory_usage'] for r in records])
            avg_efficiency = np.mean([r['efficiency'] for r in records])
            
            # Score: lower time + lower memory + higher efficiency (normalized)
            time_score = 1.0 / (avg_time + 1e-6)
            memory_score = 1.0 / (avg_memory + 1e-6)
            efficiency_score = avg_efficiency
            
            strategy_scores[strategy] = time_score * 0.4 + memory_score * 0.3 + efficiency_score * 0.3
            
        return max(strategy_scores.keys(), key=lambda x: strategy_scores[x])
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations."""
        if len(self.batch_history) < 10:
            return {'message': 'Insufficient data for recommendations'}
            
        recent_batches = list(self.batch_history)[-100:]
        
        # Analyze patterns
        avg_batch_size = np.mean([b['batch_size'] for b in recent_batches])
        avg_max_length = np.mean([b['max_length'] for b in recent_batches])
        avg_processing_time = np.mean([b['processing_time'] for b in recent_batches])
        
        recommendations = {
            'current_performance': {
                'avg_batch_size': avg_batch_size,
                'avg_max_length': avg_max_length,
                'avg_processing_time': avg_processing_time
            },
            'best_strategy': self.get_best_strategy(),
            'suggestions': []
        }
        
        # Generate suggestions
        if avg_batch_size < 8:
            recommendations['suggestions'].append(
                "Consider increasing max_batch_size for better GPU utilization"
            )
            
        if avg_processing_time > 0.1:
            recommendations['suggestions'].append(
                "Processing time is high - consider memory_aware batching or smaller batches"
            )
            
        return recommendations


if __name__ == "__main__":
    # Demo usage
    print("🔄 Dynamic Batching Demo")
    print("=" * 50)
    
    # Create mock data with variable sequence lengths
    sequence_lengths = [50, 100, 150, 200, 250, 300, 80, 120, 180, 220, 280, 320]
    mock_requests = [f"request_{i}" for i in range(len(sequence_lengths))]
    items = list(zip(mock_requests, sequence_lengths))
    
    print(f"Input sequences: {sequence_lengths}")
    
    # Test different batching strategies
    strategies = [
        BatchingStrategy.NAIVE,
        BatchingStrategy.BUCKETED,
        BatchingStrategy.MEMORY_AWARE
    ]
    
    # Mock memory estimator
    memory_estimator = MemoryEstimator(
        model_hidden_size=768,
        num_layers=12,
        num_heads=12
    )
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy.value.upper()} strategy ---")
        
        config = BatchingConfig(
            strategy=strategy,
            max_batch_size=4,
            bucket_size=64,
            memory_limit_mb=1024
        )
        
        batcher = DynamicBatcher(config, memory_estimator)
        batcher.add_items(items.copy())
        
        batch_count = 0
        while True:
            batch_items, batch_info = batcher.get_next_batch()
            if not batch_items:
                break
                
            batch_count += 1
            lengths = batch_info.get('lengths', [])
            efficiency_info = calculate_padding_efficiency(lengths)
            
            print(f"  Batch {batch_count}:")
            print(f"    Items: {len(batch_items)}")
            print(f"    Lengths: {lengths}")
            print(f"    Max length: {batch_info.get('max_length', 0)}")
            print(f"    Padding efficiency: {efficiency_info.get('efficiency', 0):.2%}")
            
            if 'estimated_memory_mb' in batch_info:
                print(f"    Est. memory: {batch_info['estimated_memory_mb']} MB")
                
    # Test continuous batching
    print(f"\n--- Testing CONTINUOUS batching ---")
    
    config = BatchingConfig(
        strategy=BatchingStrategy.CONTINUOUS,
        enable_continuous_batching=True,
        max_batch_size=6
    )
    
    batcher = DynamicBatcher(config, memory_estimator)
    
    # Simulate streaming requests
    for i in range(0, len(items), 3):
        batch_items = items[i:i+3]
        batcher.add_items(batch_items)
        
        batch, batch_info = batcher.get_next_batch()
        if batch:
            print(f"  Continuous batch:")
            print(f"    Items: {len(batch)}")
            print(f"    Continuing: {batch_info.get('continuing_requests', 0)}")
            print(f"    New: {batch_info.get('new_requests', 0)}")
    
    # Test padding function
    print(f"\n--- Testing padding function ---")
    
    input_tensors = [
        torch.randint(1, 1000, (length,)) for length in sequence_lengths[:4]
    ]
    
    padded_batch = create_padded_batch(
        input_tensors,
        padding_token_id=0,
        max_length=300
    )
    
    print(f"Padded batch shape: {padded_batch['input_ids'].shape}")
    print(f"Attention mask shape: {padded_batch['attention_mask'].shape}")
    
    # Test optimization
    print(f"\n--- Testing batch optimization ---")
    
    optimizer = BatchingOptimizer()
    
    # Simulate some batch records
    for i in range(20):
        batch_info = {
            'strategy': np.random.choice(['naive', 'bucketed', 'memory_aware']),
            'batch_size': np.random.randint(2, 8),
            'max_length': np.random.randint(100, 500),
            'efficiency': np.random.uniform(0.6, 0.95)
        }
        
        processing_time = np.random.uniform(0.01, 0.1)
        memory_usage = np.random.randint(100, 1000)
        
        optimizer.record_batch(batch_info, processing_time, memory_usage)
        
    recommendations = optimizer.get_recommendations()
    print(f"Best strategy: {recommendations.get('best_strategy', 'unknown')}")
    print(f"Suggestions: {recommendations.get('suggestions', [])}")
    
    print("\n✅ Dynamic batching demo completed!")
    print("Key features demonstrated:")
    print("- Multiple batching strategies")
    print("- Memory-aware batch formation")
    print("- Continuous batching support")
    print("- Padding efficiency optimization")
    print("- Performance-based strategy selection")