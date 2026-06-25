"""
Copyright (c) 2025. All rights reserved.
"""

"""
Comprehensive tests for the high-performance inference engine.
"""

import unittest
import torch
import torch.nn as nn
import time
import threading
import asyncio
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, MagicMock, patch

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import (
    InferenceEngine,
    InferenceRequest,
    RequestScheduler,
    BatchProcessor,
    KVCacheManager,
    PerformanceMonitor,
    EngineConfig,
    GenerationConfig,
    RequestPriority,
    RequestStatus
)
from inference.dynamic_batching import (
    DynamicBatcher,
    MemoryEstimator,
    SequenceBucketer,
    BatchingConfig,
    BatchingStrategy,
    create_padded_batch,
    calculate_padding_efficiency
)
from inference.performance_optimizer import (
    PerformanceOptimizer,
    SpeculativeDecoder,
    ParallelSampler,
    MemoryPool,
    AdaptiveInferenceController,
    OptimizationConfig,
    OptimizationLevel
)


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        x = self.embedding(input_ids)
        
        # Simple forward pass
        for layer in self.layers:
            x = torch.relu(layer(x))
            
        logits = self.output_projection(x)
        
        # Mock output structure
        class Output:
            def __init__(self, logits, past_key_values=None):
                self.logits = logits
                self.past_key_values = past_key_values if use_cache else None
                
        return Output(logits, past_key_values)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0
        self.eos_token_id = 2
        
    def encode(self, text, return_tensors=None):
        # Simple tokenization - convert to word count + some randomness
        tokens = torch.randint(3, self.vocab_size, (1, max(5, len(text.split()) + 2)))
        return tokens
        
    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        return f"Generated text with {len(tokens)} tokens"


class TestInferenceRequest(unittest.TestCase):
    """Test InferenceRequest class."""
    
    def test_request_creation(self):
        """Test request creation with default values."""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        request = InferenceRequest(
            request_id="test-001",
            input_ids=input_ids
        )
        
        self.assertEqual(request.request_id, "test-001")
        self.assertEqual(request.status, RequestStatus.QUEUED)
        self.assertEqual(request.priority, RequestPriority.NORMAL)
        self.assertIsNotNone(request.created_at)
        
    def test_request_priority_ordering(self):
        """Test that requests are ordered by priority correctly."""
        req1 = InferenceRequest("req1", torch.tensor([[1]]), priority=RequestPriority.LOW)
        req2 = InferenceRequest("req2", torch.tensor([[1]]), priority=RequestPriority.HIGH)
        
        # Higher priority should be "less than" for min-heap behavior
        self.assertTrue(req2 < req1)
        
    def test_request_with_custom_config(self):
        """Test request with custom generation config."""
        gen_config = GenerationConfig(max_length=100, temperature=0.8)
        request = InferenceRequest(
            request_id="test-002",
            input_ids=torch.tensor([[1, 2, 3]]),
            generation_config=gen_config
        )
        
        self.assertEqual(request.generation_config.max_length, 100)
        self.assertEqual(request.generation_config.temperature, 0.8)


class TestRequestScheduler(unittest.TestCase):
    """Test RequestScheduler class."""
    
    def setUp(self):
        self.config = EngineConfig(max_batch_size=4, max_queue_size=10)
        self.scheduler = RequestScheduler(self.config)
        
    def test_submit_request(self):
        """Test submitting a request."""
        request = InferenceRequest("test-001", torch.tensor([[1, 2, 3]]))
        request_id = self.scheduler.submit_request(request)
        
        self.assertEqual(request_id, "test-001")
        self.assertEqual(len(self.scheduler.request_queue), 1)
        
    def test_get_batch(self):
        """Test getting a batch of requests."""
        requests = [
            InferenceRequest(f"req-{i}", torch.tensor([[i+1]]))
            for i in range(3)
        ]
        
        for req in requests:
            self.scheduler.submit_request(req)
            
        batch = self.scheduler.get_batch(max_batch_size=2)
        
        self.assertEqual(len(batch), 2)
        self.assertEqual(len(self.scheduler.request_queue), 1)
        
        for req in batch:
            self.assertEqual(req.status, RequestStatus.PROCESSING)
            
    def test_priority_ordering(self):
        """Test that higher priority requests are processed first."""
        low_req = InferenceRequest("low", torch.tensor([[1]]), priority=RequestPriority.LOW)
        high_req = InferenceRequest("high", torch.tensor([[2]]), priority=RequestPriority.HIGH)
        
        self.scheduler.submit_request(low_req)
        self.scheduler.submit_request(high_req)
        
        batch = self.scheduler.get_batch(max_batch_size=1)
        
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].request_id, "high")
        
    def test_complete_request(self):
        """Test completing a request."""
        request = InferenceRequest("test-001", torch.tensor([[1, 2, 3]]))
        self.scheduler.submit_request(request)
        
        batch = self.scheduler.get_batch()
        request = batch[0]
        
        self.scheduler.complete_request(request)
        
        self.assertEqual(request.status, RequestStatus.COMPLETED)
        self.assertIsNotNone(request.processing_end)
        self.assertIn(request.request_id, self.scheduler.completed_requests)


class TestBatchProcessor(unittest.TestCase):
    """Test BatchProcessor class."""
    
    def setUp(self):
        self.config = EngineConfig()
        self.processor = BatchProcessor(self.config)
        
    def test_create_batch(self):
        """Test creating a padded batch."""
        requests = [
            InferenceRequest("req1", torch.tensor([[1, 2, 3]])),
            InferenceRequest("req2", torch.tensor([[4, 5]])),
            InferenceRequest("req3", torch.tensor([[6, 7, 8, 9]]))
        ]
        
        batch_data = self.processor.create_batch(requests)
        
        self.assertEqual(batch_data['batch_size'], 3)
        self.assertEqual(batch_data['input_ids'].shape, (3, 4))  # Max length is 4
        self.assertEqual(batch_data['attention_mask'].shape, (3, 4))
        
        # Check padding
        self.assertEqual(batch_data['input_ids'][1, -2:].tolist(), [0, 0])  # Last two should be padding
        
    def test_split_batch_by_length(self):
        """Test splitting batch by sequence length."""
        requests = [
            InferenceRequest("short1", torch.tensor([[1, 2]])),
            InferenceRequest("short2", torch.tensor([[3, 4]])),
            InferenceRequest("long1", torch.tensor([[5, 6, 7, 8, 9, 10]])),
            InferenceRequest("long2", torch.tensor([[11, 12, 13, 14, 15, 16]]))
        ]
        
        batches = self.processor.split_batch_by_length(requests)
        
        self.assertEqual(len(batches), 2)  # Should split into short and long batches
        
        # Check that similar lengths are grouped
        short_batch = [req for req in batches[0] if req.input_ids.size(-1) <= 3]
        long_batch = [req for req in batches[1] if req.input_ids.size(-1) > 3]
        
        self.assertTrue(len(short_batch) > 0)
        self.assertTrue(len(long_batch) > 0)


class TestKVCacheManager(unittest.TestCase):
    """Test KVCacheManager class."""
    
    def setUp(self):
        self.config = EngineConfig(kv_cache_max_size=3)
        self.cache_manager = KVCacheManager(self.config)
        
    def test_store_and_get_cache(self):
        """Test storing and retrieving KV cache."""
        kv_cache = {
            'key': torch.randn(2, 4, 64),
            'value': torch.randn(2, 4, 64)
        }
        
        self.cache_manager.store_cache("req-001", kv_cache)
        retrieved = self.cache_manager.get_cache("req-001")
        
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(retrieved['key'], kv_cache['key']))
        
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(4):  # One more than max_size
            kv_cache = {'key': torch.randn(1, 2, 32), 'value': torch.randn(1, 2, 32)}
            self.cache_manager.store_cache(f"req-{i:03d}", kv_cache)
            
        # First request should be evicted
        self.assertIsNone(self.cache_manager.get_cache("req-000"))
        self.assertIsNotNone(self.cache_manager.get_cache("req-003"))
        
    def test_clear_cache(self):
        """Test clearing cache."""
        kv_cache = {'key': torch.randn(1, 2, 32), 'value': torch.randn(1, 2, 32)}
        self.cache_manager.store_cache("req-001", kv_cache)
        
        self.cache_manager.clear_cache("req-001")
        self.assertIsNone(self.cache_manager.get_cache("req-001"))


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor class."""
    
    def setUp(self):
        self.config = EngineConfig(enable_metrics=True)
        self.monitor = PerformanceMonitor(self.config)
        
    def test_record_metric(self):
        """Test recording metrics."""
        self.monitor.record_metric("latency", 0.05)
        self.monitor.record_metric("latency", 0.03)
        self.monitor.record_metric("throughput", 150.0)
        
        metrics = self.monitor.get_metrics("latency")
        self.assertEqual(len(metrics["latency"]), 2)
        
    def test_summary_stats(self):
        """Test getting summary statistics."""
        values = [0.01, 0.02, 0.03, 0.04, 0.05]
        for val in values:
            self.monitor.record_metric("test_metric", val)
            
        stats = self.monitor.get_summary_stats()
        
        self.assertIn("test_metric", stats)
        self.assertEqual(stats["test_metric"]["count"], 5)
        self.assertAlmostEqual(stats["test_metric"]["mean"], 0.03, places=3)
        self.assertEqual(stats["test_metric"]["min"], 0.01)
        self.assertEqual(stats["test_metric"]["max"], 0.05)


class TestDynamicBatching(unittest.TestCase):
    """Test dynamic batching components."""
    
    def test_memory_estimator(self):
        """Test memory usage estimation."""
        estimator = MemoryEstimator(
            model_hidden_size=768,
            num_layers=12,
            num_heads=12
        )
        
        memory_usage = estimator.estimate_batch_memory(
            batch_size=4,
            sequence_length=512
        )
        
        self.assertGreater(memory_usage, 0)
        
        # Larger batch should use more memory
        larger_memory = estimator.estimate_batch_memory(
            batch_size=8,
            sequence_length=512
        )
        
        self.assertGreater(larger_memory, memory_usage)
        
    def test_sequence_bucketer(self):
        """Test sequence bucketing."""
        bucketer = SequenceBucketer(bucket_size=64)
        
        # Add sequences of different lengths
        sequences = [
            ("seq1", 50),
            ("seq2", 60),
            ("seq3", 120),
            ("seq4", 130)
        ]
        
        for seq, length in sequences:
            bucketer.add_sequence(seq, length)
            
        # Should have 2 buckets
        self.assertEqual(len(bucketer.buckets), 2)
        
        # Get batch from fullest bucket
        batch_items, bucket_length = bucketer.get_batch_from_bucket(max_batch_size=3)
        
        self.assertGreater(len(batch_items), 0)
        self.assertGreater(bucket_length, 0)
        
    def test_dynamic_batcher_strategies(self):
        """Test different batching strategies."""
        items = [("item1", 100), ("item2", 150), ("item3", 200), ("item4", 120)]
        
        # Test naive strategy
        config = BatchingConfig(strategy=BatchingStrategy.NAIVE, max_batch_size=2)
        batcher = DynamicBatcher(config)
        batcher.add_items(items.copy())
        
        batch_items, batch_info = batcher.get_next_batch()
        self.assertEqual(len(batch_items), 2)
        self.assertEqual(batch_info['strategy'], 'naive')
        
        # Test bucketed strategy
        config = BatchingConfig(strategy=BatchingStrategy.BUCKETED, bucket_size=64)
        batcher = DynamicBatcher(config)
        batcher.add_items(items.copy())
        
        batch_items, batch_info = batcher.get_next_batch()
        self.assertGreater(len(batch_items), 0)
        self.assertEqual(batch_info['strategy'], 'bucketed')
        
    def test_create_padded_batch(self):
        """Test creating padded batch from tensors."""
        input_ids_list = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5]]),
            torch.tensor([[6, 7, 8, 9]])
        ]
        
        padded_batch = create_padded_batch(
            input_ids_list,
            padding_token_id=0,
            max_length=4
        )
        
        self.assertEqual(padded_batch['input_ids'].shape, (3, 4))
        self.assertEqual(padded_batch['attention_mask'].shape, (3, 4))
        
        # Check padding is correct
        self.assertEqual(padded_batch['input_ids'][1, -2:].tolist(), [0, 0])
        
    def test_padding_efficiency(self):
        """Test padding efficiency calculation."""
        sequence_lengths = [10, 15, 20, 12]
        
        efficiency = calculate_padding_efficiency(sequence_lengths)
        
        self.assertIn('efficiency', efficiency)
        self.assertIn('waste_ratio', efficiency)
        self.assertGreater(efficiency['efficiency'], 0)
        self.assertLess(efficiency['efficiency'], 1)
        
        # Total tokens + padding tokens should equal total padded tokens
        total_padded = efficiency['total_tokens'] + efficiency['padding_tokens']
        expected_padded = len(sequence_lengths) * max(sequence_lengths)
        self.assertEqual(total_padded, expected_padded)


class TestPerformanceOptimizer(unittest.TestCase):
    """Test performance optimization components."""
    
    def setUp(self):
        self.model = MockModel()
        
    def test_memory_pool(self):
        """Test memory pool operations."""
        device = torch.device('cpu')
        pool = MemoryPool(pool_size=1024*1024, device=device)  # 1MB
        
        # Get tensor from pool
        tensor1 = pool.get_tensor((10, 20), torch.float32)
        self.assertEqual(tensor1.shape, (10, 20))
        
        # Return tensor to pool
        pool.return_tensor(tensor1)
        self.assertGreater(pool.allocated_size, 0)
        
        # Get another tensor (should reuse from pool)
        tensor2 = pool.get_tensor((10, 20), torch.float32)
        self.assertEqual(tensor2.shape, (10, 20))
        
    def test_adaptive_inference_controller(self):
        """Test adaptive parameter control."""
        controller = AdaptiveInferenceController()
        
        # Update with some metrics
        for i in range(15):
            latency = 0.1 + i * 0.01  # Increasing latency
            memory_usage = 0.5
            throughput = 100 - i * 5  # Decreasing throughput
            
            controller.update_metrics(latency, memory_usage, throughput)
            
        # Get adapted parameters
        params = controller.adapt_parameters()
        
        self.assertIn('batch_size', params)
        self.assertIn('temperature', params)
        self.assertIn('speculation_length', params)
        
        # High latency should increase speculation length
        self.assertGreater(params['speculation_length'], 4)
        
    def test_speculative_decoder(self):
        """Test speculative decoding (basic functionality)."""
        main_model = self.model
        decoder = SpeculativeDecoder(main_model)
        
        input_ids = torch.tensor([[1, 2, 3, 4]])
        
        # Test single token generation (fallback)
        next_token = decoder._generate_single_token(input_ids, temperature=1.0)
        self.assertEqual(next_token.shape, (1, 1))
        
        # Test acceptance rate tracking
        initial_rate = decoder.get_acceptance_rate()
        self.assertEqual(initial_rate, 0.0)  # No history yet
        
    def test_parallel_sampler(self):
        """Test parallel sampling."""
        sampler = ParallelSampler(self.model, num_parallel=2)
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Test single sequence generation
        output = sampler._generate_single_sequence(
            input_ids, max_length=10, temperature=1.0
        )
        
        self.assertGreater(output.size(1), input_ids.size(1))
        
    def test_performance_optimizer_integration(self):
        """Test full performance optimizer."""
        config = OptimizationConfig(
            level=OptimizationLevel.BASIC,
            enable_speculative_decoding=False,  # Disable for simpler test
            enable_parallel_sampling=False,
            compile_model=False
        )
        
        optimizer = PerformanceOptimizer(self.model, config)
        
        input_ids = torch.tensor([[1, 2, 3, 4]])
        
        # Test optimized generation
        output, stats = optimizer.optimized_generate(
            input_ids,
            max_length=15,
            temperature=1.0
        )
        
        self.assertGreater(output.size(1), input_ids.size(1))
        self.assertIn('strategy', stats)
        self.assertIn('generation_time', stats)
        self.assertIn('tokens_generated', stats)
        
        # Get optimization stats
        opt_stats = optimizer.get_optimization_stats()
        self.assertIn('memory_usage', opt_stats)


class TestInferenceEngineIntegration(unittest.TestCase):
    """Test full inference engine integration."""
    
    def setUp(self):
        self.model = MockModel()
        self.tokenizer = MockTokenizer()
        self.config = EngineConfig(
            max_batch_size=2,
            batch_timeout_ms=10,  # Short timeout for testing
            enable_kv_cache=False,  # Simplify for testing
            enable_metrics=True
        )
        
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = InferenceEngine(
            self.model,
            self.tokenizer,
            self.config
        )
        
        self.assertEqual(engine.config.max_batch_size, 2)
        self.assertIsNotNone(engine.scheduler)
        self.assertIsNotNone(engine.batch_processor)
        
    def test_submit_and_process_request(self):
        """Test submitting and processing requests."""
        engine = InferenceEngine(
            self.model,
            self.tokenizer,
            self.config
        )
        
        # Start engine
        engine.start()
        
        try:
            # Submit request
            request_id = engine.submit_request(
                "Hello world",
                generation_config=GenerationConfig(max_length=20)
            )
            
            self.assertIsNotNone(request_id)
            
            # Wait for processing
            time.sleep(0.5)
            
            # Check request status
            status = engine.scheduler.get_request_status(request_id)
            self.assertIn(status, [RequestStatus.PROCESSING, RequestStatus.COMPLETED])
            
        finally:
            engine.stop()
            
    def test_multiple_requests_batching(self):
        """Test batching multiple requests."""
        engine = InferenceEngine(
            self.model,
            self.tokenizer,
            self.config
        )
        
        engine.start()
        
        try:
            # Submit multiple requests
            request_ids = []
            for i in range(3):
                request_id = engine.submit_request(
                    f"Test request {i}",
                    generation_config=GenerationConfig(max_length=15)
                )
                request_ids.append(request_id)
                
            # Wait for processing
            time.sleep(1.0)
            
            # Check that requests were processed
            for request_id in request_ids:
                status = engine.scheduler.get_request_status(request_id)
                self.assertIsNotNone(status)
                
        finally:
            engine.stop()
            
    def test_priority_handling(self):
        """Test request priority handling."""
        engine = InferenceEngine(
            self.model,
            self.tokenizer,
            self.config
        )
        
        engine.start()
        
        try:
            # Submit low priority request first
            low_priority_id = engine.submit_request(
                "Low priority request",
                priority=RequestPriority.LOW
            )
            
            # Then high priority request
            high_priority_id = engine.submit_request(
                "High priority request",
                priority=RequestPriority.HIGH
            )
            
            # Wait for processing
            time.sleep(0.5)
            
            # High priority should be processed
            high_status = engine.scheduler.get_request_status(high_priority_id)
            self.assertIn(high_status, [RequestStatus.PROCESSING, RequestStatus.COMPLETED])
            
        finally:
            engine.stop()
            
    def test_engine_statistics(self):
        """Test engine statistics collection."""
        engine = InferenceEngine(
            self.model,
            self.tokenizer,
            self.config
        )
        
        engine.start()
        
        try:
            # Submit a request
            engine.submit_request("Test for stats")
            
            # Wait for processing
            time.sleep(0.5)
            
            # Get statistics
            stats = engine.get_stats()
            
            self.assertIn('engine_status', stats)
            self.assertIn('device', stats)
            self.assertIn('performance_metrics', stats)
            self.assertEqual(stats['engine_status'], 'running')
            
        finally:
            engine.stop()
            
    @patch('asyncio.Future')
    def test_async_request_submission(self):
        """Test async request submission."""
        engine = InferenceEngine(
            self.model,
            self.tokenizer,
            self.config
        )
        
        # Test that async submission creates proper callback
        try:
            # This would normally be run in an async context
            request_id = engine.submit_request(
                "Async test",
                callback=lambda req: None  # Mock callback
            )
            
            self.assertIsNotNone(request_id)
            
        finally:
            if engine.is_running:
                engine.stop()


class TestBenchmarks(unittest.TestCase):
    """Benchmark tests for performance validation."""
    
    def setUp(self):
        self.model = MockModel()
        self.tokenizer = MockTokenizer()
        
    def test_throughput_benchmark(self):
        """Test throughput under load."""
        config = EngineConfig(
            max_batch_size=4,
            batch_timeout_ms=50,
            enable_metrics=True
        )
        
        engine = InferenceEngine(self.model, self.tokenizer, config)
        engine.start()
        
        try:
            start_time = time.time()
            request_ids = []
            
            # Submit many requests
            for i in range(10):
                request_id = engine.submit_request(
                    f"Benchmark request {i}",
                    generation_config=GenerationConfig(max_length=20)
                )
                request_ids.append(request_id)
                
            # Wait for all to complete
            time.sleep(2.0)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate throughput
            completed_count = sum(
                1 for req_id in request_ids
                if engine.scheduler.get_request_status(req_id) == RequestStatus.COMPLETED
            )
            
            throughput = completed_count / total_time
            print(f"Throughput: {throughput:.2f} requests/second")
            
            # Basic assertion - should complete at least some requests
            self.assertGreater(completed_count, 0)
            
        finally:
            engine.stop()
            
    def test_latency_benchmark(self):
        """Test latency for single requests."""
        config = EngineConfig(max_batch_size=1)  # Force single request processing
        
        engine = InferenceEngine(self.model, self.tokenizer, config)
        engine.start()
        
        latencies = []
        
        try:
            for i in range(5):
                start_time = time.time()
                
                request_id = engine.submit_request(
                    "Latency test request",
                    generation_config=GenerationConfig(max_length=10)
                )
                
                # Wait for completion
                while True:
                    status = engine.scheduler.get_request_status(request_id)
                    if status == RequestStatus.COMPLETED:
                        break
                    time.sleep(0.01)
                    
                latency = time.time() - start_time
                latencies.append(latency)
                
            avg_latency = sum(latencies) / len(latencies)
            print(f"Average latency: {avg_latency:.3f} seconds")
            
            # Basic assertion - latency should be reasonable
            self.assertLess(avg_latency, 5.0)  # Should complete within 5 seconds
            
        finally:
            engine.stop()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)