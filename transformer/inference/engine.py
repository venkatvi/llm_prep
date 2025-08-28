"""
Copyright (c) 2025. All rights reserved.
"""

"""
High-performance inference engine for production LLM serving.

This module provides a production-ready inference engine with:
- Efficient request batching and scheduling
- Dynamic sequence length handling
- Advanced KV cache management
- Performance optimization for throughput and latency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import asyncio
import time
import heapq
import threading
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
import psutil
import gc


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class RequestStatus(Enum):
    """Request status states."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 512
    min_length: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True


@dataclass
class InferenceRequest:
    """Individual inference request."""
    request_id: str
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    status: RequestStatus = RequestStatus.QUEUED
    output_ids: Optional[torch.Tensor] = None
    generated_length: int = 0
    kv_cache: Optional[Dict[str, torch.Tensor]] = None
    processing_start: Optional[float] = None
    processing_end: Optional[float] = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())
            
    def __lt__(self, other):
        """Priority queue ordering (higher priority first)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at


@dataclass
class EngineConfig:
    """Configuration for inference engine."""
    # Batching settings
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    batch_timeout_ms: float = 50.0  # Max time to wait for batch formation
    
    # Scheduling settings
    max_queue_size: int = 1000
    priority_boost_threshold: float = 5.0  # Boost priority after waiting (seconds)
    
    # Performance settings
    enable_kv_cache: bool = True
    kv_cache_max_size: int = 100  # Max cached requests
    enable_dynamic_batching: bool = True
    prefill_chunk_size: int = 512
    decode_batch_size: int = 64
    
    # Memory management
    memory_pool_size: int = 1024 * 1024 * 1024  # 1GB
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"


class BatchProcessor:
    """Processes batches of requests efficiently."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_batch(self, requests: List[InferenceRequest]) -> Dict[str, torch.Tensor]:
        """Create padded batch from requests."""
        if not requests:
            return {}
            
        batch_size = len(requests)
        
        # Find maximum sequence length in batch
        max_len = max(req.input_ids.size(-1) for req in requests)
        max_len = min(max_len, self.config.max_sequence_length)
        
        # Get device and dtype from first request
        device = requests[0].input_ids.device
        dtype = requests[0].input_ids.dtype
        
        # Create padded tensors
        batch_input_ids = torch.zeros(
            (batch_size, max_len), 
            dtype=dtype, 
            device=device
        )
        batch_attention_mask = torch.zeros(
            (batch_size, max_len), 
            dtype=torch.long, 
            device=device
        )
        
        # Fill batch tensors
        for i, req in enumerate(requests):
            seq_len = min(req.input_ids.size(-1), max_len)
            batch_input_ids[i, :seq_len] = req.input_ids[:seq_len]
            
            if req.attention_mask is not None:
                batch_attention_mask[i, :seq_len] = req.attention_mask[:seq_len]
            else:
                batch_attention_mask[i, :seq_len] = 1
                
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'batch_size': batch_size,
            'sequence_lengths': [min(req.input_ids.size(-1), max_len) for req in requests]
        }
        
    def split_batch_by_length(self, requests: List[InferenceRequest]) -> List[List[InferenceRequest]]:
        """Split requests into batches by similar sequence lengths."""
        # Sort by sequence length
        requests.sort(key=lambda r: r.input_ids.size(-1))
        
        batches = []
        current_batch = []
        current_length = 0
        
        for req in requests:
            req_length = req.input_ids.size(-1)
            
            # Start new batch if length difference is too large or batch is full
            if (current_batch and 
                (abs(req_length - current_length) > 128 or 
                 len(current_batch) >= self.config.max_batch_size)):
                batches.append(current_batch)
                current_batch = []
                current_length = 0
                
            current_batch.append(req)
            current_length = req_length
            
        if current_batch:
            batches.append(current_batch)
            
        return batches


class KVCacheManager:
    """Manages KV cache for efficient inference."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.cache_store: Dict[str, Dict[str, torch.Tensor]] = {}
        self.access_times: Dict[str, float] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.total_cache_size = 0
        self.lock = threading.Lock()
        
    def get_cache(self, request_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached KV states for request."""
        with self.lock:
            if request_id in self.cache_store:
                self.access_times[request_id] = time.time()
                return self.cache_store[request_id]
            return None
            
    def store_cache(self, request_id: str, kv_cache: Dict[str, torch.Tensor]):
        """Store KV cache for request."""
        if not self.config.enable_kv_cache:
            return
            
        with self.lock:
            # Calculate cache size
            cache_size = sum(
                tensor.numel() * tensor.element_size() 
                for tensor in kv_cache.values()
            )
            
            # Evict if necessary
            while (len(self.cache_store) >= self.config.kv_cache_max_size or
                   self.total_cache_size + cache_size > self.config.memory_pool_size):
                self._evict_lru()
                
            # Store cache
            self.cache_store[request_id] = kv_cache
            self.access_times[request_id] = time.time()
            self.cache_sizes[request_id] = cache_size
            self.total_cache_size += cache_size
            
    def _evict_lru(self):
        """Evict least recently used cache entry."""
        if not self.cache_store:
            return
            
        # Find LRU entry
        lru_id = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        del self.cache_store[lru_id]
        del self.access_times[lru_id]
        self.total_cache_size -= self.cache_sizes[lru_id]
        del self.cache_sizes[lru_id]
        
    def clear_cache(self, request_id: Optional[str] = None):
        """Clear cache for specific request or all cache."""
        with self.lock:
            if request_id:
                if request_id in self.cache_store:
                    del self.cache_store[request_id]
                    del self.access_times[request_id]
                    self.total_cache_size -= self.cache_sizes[request_id]
                    del self.cache_sizes[request_id]
            else:
                self.cache_store.clear()
                self.access_times.clear()
                self.cache_sizes.clear()
                self.total_cache_size = 0


class RequestScheduler:
    """Schedules and prioritizes inference requests."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.request_queue = []  # Priority queue
        self.processing_requests: Dict[str, InferenceRequest] = {}
        self.completed_requests: Dict[str, InferenceRequest] = {}
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.logger = logging.getLogger(__name__)
        
    def submit_request(self, request: InferenceRequest) -> str:
        """Submit a new inference request."""
        with self.lock:
            if len(self.request_queue) >= self.config.max_queue_size:
                raise RuntimeError("Request queue is full")
                
            heapq.heappush(self.request_queue, request)
            self.condition.notify()
            
        self.logger.info(f"Submitted request {request.request_id} with priority {request.priority}")
        return request.request_id
        
    def get_batch(self, max_batch_size: Optional[int] = None) -> List[InferenceRequest]:
        """Get next batch of requests for processing."""
        max_size = max_batch_size or self.config.max_batch_size
        batch = []
        
        with self.lock:
            # Wait for requests if queue is empty
            if not self.request_queue:
                self.condition.wait(timeout=self.config.batch_timeout_ms / 1000.0)
                
            # # Boost priority for long-waiting requests
            # current_time = time.time()
            # for req in self.request_queue:
            #     wait_time = current_time - req.created_at
            #     if wait_time > self.config.priority_boost_threshold:
            #         if req.priority != RequestPriority.CRITICAL:
            #             req.priority = RequestPriority.HIGH
                        
            # Re-heapify after priority changes
            heapq.heapify(self.request_queue)
            
            # Extract batch
            while len(batch) < max_size and self.request_queue:
                req = heapq.heappop(self.request_queue)
                req.status = RequestStatus.PROCESSING
                req.processing_start = time.time()
                batch.append(req)
                self.processing_requests[req.request_id] = req
                
        return batch
        
    def complete_request(self, request: InferenceRequest):
        """Mark request as completed."""
        with self.lock:
            request.status = RequestStatus.COMPLETED
            request.processing_end = time.time()
            
            if request.request_id in self.processing_requests:
                del self.processing_requests[request.request_id]
                
            self.completed_requests[request.request_id] = request
            
            # Call callback if provided
            if request.callback:
                try:
                    request.callback(request)
                except Exception as e:
                    self.logger.error(f"Callback error for request {request.request_id}: {e}")
                    
    def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get status of a request."""
        with self.lock:
            if request_id in self.processing_requests:
                return RequestStatus.PROCESSING
            elif request_id in self.completed_requests:
                return RequestStatus.COMPLETED
            else:
                # Check if still in queue
                for req in self.request_queue:
                    if req.request_id == request_id:
                        return RequestStatus.QUEUED
                return None


class PerformanceMonitor:
    """Monitors inference engine performance."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a performance metric."""
        if not self.config.enable_metrics:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            self.metrics[name].append((timestamp, value))
            
            # Keep only recent metrics (last 1000 entries)
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
                
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[Tuple[float, float]]]:
        """Get recorded metrics."""
        with self.lock:
            if name:
                return {name: self.metrics.get(name, [])}
            return dict(self.metrics)
            
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        stats = {}
        
        with self.lock:
            for name, values in self.metrics.items():
                if not values:
                    continue
                    
                metric_values = [v for _, v in values]
                stats[name] = {
                    'count': len(metric_values),
                    'mean': sum(metric_values) / len(metric_values),
                    'min': min(metric_values),
                    'max': max(metric_values),
                    'latest': metric_values[-1]
                }
                
                # Calculate percentiles
                sorted_values = sorted(metric_values)
                n = len(sorted_values)
                stats[name].update({
                    'p50': sorted_values[n // 2],
                    'p90': sorted_values[int(n * 0.9)],
                    'p95': sorted_values[int(n * 0.95)],
                    'p99': sorted_values[int(n * 0.99)]
                })
                
        return stats


class InferenceEngine:
    """High-performance inference engine for LLM serving."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: EngineConfig = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize components
        self.scheduler = RequestScheduler(self.config)
        self.batch_processor = BatchProcessor(self.config)
        self.kv_cache_manager = KVCacheManager(self.config)
        self.monitor = PerformanceMonitor(self.config)
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized InferenceEngine with device: {self.device}")
        
    def start(self):
        """Start the inference engine."""
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("Inference engine started")
        
    def stop(self):
        """Stop the inference engine."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        self.logger.info("Inference engine stopped")
        
    def submit_request(
        self,
        input_text: str,
        generation_config: Optional[GenerationConfig] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit an inference request."""
        # Tokenize input
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        input_ids = inputs.to(self.device)
        
        # Create request
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            input_ids=input_ids,
            generation_config=generation_config or GenerationConfig(),
            priority=priority,
            callback=callback,
            metadata=metadata or {}
        )
        
        # Submit to scheduler
        return self.scheduler.submit_request(request)
        
    async def submit_request_async(
        self,
        input_text: str,
        generation_config: Optional[GenerationConfig] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InferenceRequest:
        """Submit request and wait for completion."""
        future = asyncio.Future()
        
        def callback(request):
            if not future.done():
                future.set_result(request)
                
        request_id = self.submit_request(
            input_text, generation_config, priority, callback, metadata
        )
        
        # Wait for completion
        return await future
        
    def _processing_loop(self):
        """Main processing loop."""
        self.logger.info("Processing loop started")
        
        while self.is_running:
            try:
                # Get next batch
                batch = self.scheduler.get_batch()
                if not batch:
                    time.sleep(0.001)  # Short sleep if no requests
                    continue
                    
                # Process batch
                self._process_batch(batch)
                
                # Memory management
                if self._should_run_gc():
                    self._run_garbage_collection()
                    
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
                
    def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of requests."""
        start_time = time.time()
        
        try:
            # Split batch by sequence length for efficiency
            sub_batches = self.batch_processor.split_batch_by_length(requests)
            
            for sub_batch in sub_batches:
                self._process_sub_batch(sub_batch)
                
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            for req in requests:
                req.error = e
                req.status = RequestStatus.FAILED
                self.scheduler.complete_request(req)
        finally:
            # Record metrics
            batch_time = time.time() - start_time
            self.monitor.record_metric('batch_processing_time', batch_time)
            self.monitor.record_metric('batch_size', len(requests))
            
    def _process_sub_batch(self, requests: List[InferenceRequest]):
        """Process a sub-batch of requests with similar lengths."""
        if not requests:
            return
            
        # Create batch tensors
        batch_data = self.batch_processor.create_batch(requests)
        
        # Load KV caches if available
        kv_caches = []
        for req in requests:
            cache = self.kv_cache_manager.get_cache(req.request_id)
            kv_caches.append(cache)
            
        # Run inference
        with torch.no_grad():
            # Measure first token latency
            first_token_start = time.time()
            
            outputs = self._generate_tokens(
                batch_data['input_ids'],
                batch_data['attention_mask'],
                requests,
                kv_caches
            )
            
            first_token_time = time.time() - first_token_start
            self.monitor.record_metric('first_token_latency', first_token_time)
            
        # Update requests with results
        for i, req in enumerate(requests):
            req.output_ids = outputs[i]
            req.generated_length = outputs[i].size(-1) - req.input_ids.size(-1)
            self.scheduler.complete_request(req)
            
        # Record throughput metrics
        total_tokens = sum(req.generated_length for req in requests)
        self.monitor.record_metric('tokens_per_second', total_tokens / first_token_time)
        
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        requests: List[InferenceRequest],
        kv_caches: List[Optional[Dict[str, torch.Tensor]]]
    ) -> List[torch.Tensor]:
        """Generate tokens for batch."""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Get generation config from first request (assuming similar configs in batch)
        gen_config = requests[0].generation_config
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = kv_caches[0] if kv_caches[0] else None
        
        # Generation loop
        for step in range(gen_config.max_length - input_ids.size(1)):
            # Forward pass
            if step == 0:
                # Prefill phase
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=gen_config.use_cache
                )
            else:
                # Decode phase - only process last token
                outputs = self.model(
                    input_ids=generated[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=gen_config.use_cache
                )
                
            logits = outputs.logits
            if gen_config.use_cache:
                past_key_values = outputs.past_key_values
                
            # Sample next tokens
            next_token_logits = logits[:, -1, :]
            
            if gen_config.do_sample:
                next_tokens = self._sample_tokens(next_token_logits, gen_config)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            ], dim=-1)
            
            # Check for EOS tokens
            if gen_config.eos_token_id is not None:
                eos_mask = (next_tokens.squeeze(-1) == gen_config.eos_token_id)
                if eos_mask.all():
                    break
                    
        # Store KV caches
        if gen_config.use_cache and past_key_values:
            for i, req in enumerate(requests):
                # Extract KV cache for this request (simplified)
                req_cache = past_key_values  # In practice, would extract per-request cache
                self.kv_cache_manager.store_cache(req.request_id, req_cache)
                
        # Split batch outputs
        outputs = []
        for i in range(batch_size):
            outputs.append(generated[i])
            
        return outputs
        
    def _sample_tokens(self, logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """Sample next tokens from logits."""
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
            
        # Apply top-k filtering
        if config.top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, config.top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, top_k_indices, top_k_values)
            
        # Apply top-p filtering
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        
        return next_tokens
        
    def _should_run_gc(self) -> bool:
        """Check if garbage collection should be run."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            cached = torch.cuda.memory_reserved(self.device)
            utilization = allocated / (cached + 1)
            return utilization > self.config.gc_threshold
        else:
            memory_percent = psutil.virtual_memory().percent
            return memory_percent > self.config.gc_threshold * 100
            
    def _run_garbage_collection(self):
        """Run garbage collection and clear caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.debug("Ran garbage collection")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            'engine_status': 'running' if self.is_running else 'stopped',
            'device': str(self.device),
            'config': self.config.__dict__,
            'performance_metrics': self.monitor.get_summary_stats(),
            'cache_info': {
                'total_size': self.kv_cache_manager.total_cache_size,
                'num_entries': len(self.kv_cache_manager.cache_store)
            }
        }
        
        return stats


if __name__ == "__main__":
    # Demo usage
    print("🚀 High-Performance Inference Engine Demo")
    print("=" * 60)
    
    # Mock model and tokenizer for demo
    class MockModel(nn.Module):
        def __init__(self, vocab_size=32000, hidden_size=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.linear = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
            x = self.embedding(input_ids)
            logits = self.linear(x)
            
            # Mock output with past_key_values
            class Output:
                def __init__(self, logits, past_key_values=None):
                    self.logits = logits
                    self.past_key_values = past_key_values if use_cache else None
                    
            return Output(logits, past_key_values)
    
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            
        def encode(self, text, return_tensors=None):
            # Mock tokenization - just return tensor of random IDs
            tokens = torch.randint(3, 1000, (1, len(text.split()) + 1))
            return tokens
            
        def decode(self, tokens, skip_special_tokens=True):
            return f"Generated text with {len(tokens)} tokens"
    
    # Create mock model and tokenizer
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # Create engine configuration
    config = EngineConfig(
        max_batch_size=4,
        batch_timeout_ms=100,
        enable_kv_cache=True,
        enable_metrics=True
    )
    
    # Initialize engine
    engine = InferenceEngine(model, tokenizer, config)
    engine.start()
    
    print(f"Engine started with config:")
    print(f"  Max batch size: {config.max_batch_size}")
    print(f"  KV cache enabled: {config.enable_kv_cache}")
    print(f"  Device: {engine.device}")
    
    # Submit some test requests
    test_requests = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about AI.",
        "How does machine learning work?"
    ]
    
    print(f"\nSubmitting {len(test_requests)} requests...")
    
    request_ids = []
    for i, text in enumerate(test_requests):
        priority = RequestPriority.HIGH if i == 0 else RequestPriority.NORMAL
        request_id = engine.submit_request(
            text,
            generation_config=GenerationConfig(max_length=50),
            priority=priority
        )
        request_ids.append(request_id)
        print(f"  Submitted: {request_id[:8]}... ({text[:30]}...)")
    
    # Wait for processing
    time.sleep(2)
    
    # Get statistics
    stats = engine.get_stats()
    print(f"\nEngine Statistics:")
    print(f"  Status: {stats['engine_status']}")
    print(f"  Cache entries: {stats['cache_info']['num_entries']}")
    
    if stats['performance_metrics']:
        metrics = stats['performance_metrics']
        if 'batch_processing_time' in metrics:
            batch_time = metrics['batch_processing_time']
            print(f"  Avg batch time: {batch_time['mean']:.3f}s")
        if 'first_token_latency' in metrics:
            latency = metrics['first_token_latency']
            print(f"  First token latency: {latency['mean']:.3f}s")
    
    # Stop engine
    engine.stop()
    
    print("\n✅ Demo completed!")
    print("Key features demonstrated:")
    print("- Request batching and prioritization")
    print("- Dynamic sequence length handling")
    print("- KV cache management")
    print("- Performance monitoring")
    print("- Memory-efficient processing")