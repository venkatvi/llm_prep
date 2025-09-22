# Distributed Model Inference Pipeline - Design Discussion

## Overview
This document explores key architectural decisions for production-scale distributed ML inference using MapReduce, addressing real-world challenges in deploying ML models across distributed systems.

---

## 1. Model Loading Across Workers

### **Challenge**
Loading ML models (often GB-sized) across hundreds of worker nodes efficiently without overwhelming network bandwidth or causing startup delays.

### **Solution Strategies**

#### **A. Model Broadcasting (Hadoop/Spark Pattern)**
```python
# Master broadcasts model once to all workers
class ModelBroadcastManager:
    def __init__(self, model_path: str):
        self.model_cache = {}

    def broadcast_model(self, workers: List[str]):
        """Send model to all workers once at job start"""
        model_bytes = self._serialize_model(model_path)
        for worker in workers:
            self._send_to_worker(worker, model_bytes)
```

**Pros:** Single network transfer per worker, efficient for long-running jobs
**Cons:** Large memory footprint, slow startup for heavy models

#### **B. Distributed Model Registry (Production Approach)**
```python
class ModelRegistry:
    """Central model storage with caching"""
    def __init__(self, registry_url: str):
        self.registry = ModelStore(registry_url)

    def get_model(self, model_id: str, version: str):
        # Workers pull from shared storage (S3, GCS, etc.)
        cache_key = f"{model_id}:{version}"
        if not self._local_cache.has(cache_key):
            model = self.registry.download(model_id, version)
            self._local_cache.store(cache_key, model)
        return self._local_cache.get(cache_key)
```

**Pros:** Scalable, versioned, fault-tolerant
**Cons:** Network latency per worker startup

#### **C. Containerized Models (Modern Approach)**
```python
# Docker containers with pre-loaded models
FROM python:3.9
COPY recommendation_model_v2.1.pkl /models/
COPY inference_worker.py /app/
CMD ["python", "/app/inference_worker.py", "--model=/models/recommendation_model_v2.1.pkl"]
```

**Pros:** Immutable, reproducible, fast startup
**Cons:** Large container images, storage overhead

### **Recommendation**
Use **Model Registry** with local caching for production systems:
- Models stored in versioned object storage (S3/GCS)
- Workers cache models locally on first use
- TTL-based cache invalidation for updates

---

## 2. Handling Inference Failures

### **Challenge**
Individual product inference failures shouldn't crash entire MapReduce jobs or cause data loss.

### **Solution: Multi-Level Error Handling**

#### **A. Product-Level Resilience**
```python
def robust_inference_mapper(input_file: str, model_path: str):
    model = load_model_with_retry(model_path)
    failed_products = []

    for product in read_products(input_file):
        try:
            score, confidence = model.predict(product)
            yield (product.category, ProductScore(...))
        except InferenceError as e:
            # Log failure but continue processing
            failed_products.append({
                'product_id': product.product_id,
                'error': str(e),
                'timestamp': time.time()
            })
            # Optionally yield default score
            yield (product.category, ProductScore(
                product_id=product.product_id,
                score=0.5,  # Default neutral score
                confidence=0.0,  # Mark as low confidence
                inference_timestamp=time.time()
            ))

    # Report failures for monitoring
    if failed_products:
        report_failures(failed_products)
```

#### **B. Batch-Level Recovery**
```python
class BatchInferenceProcessor:
    def __init__(self, batch_size: int = 100, max_retries: int = 3):
        self.batch_size = batch_size
        self.max_retries = max_retries

    def process_batch(self, products: List[ProductFeatures]) -> List[ProductScore]:
        for attempt in range(self.max_retries):
            try:
                return self.model.predict_batch(products)
            except ModelTimeoutError:
                if attempt == self.max_retries - 1:
                    # Fallback to individual inference
                    return self._process_individually(products)
                time.sleep(2 ** attempt)  # Exponential backoff
```

#### **C. Dead Letter Queue Pattern**
```python
class FailedInferenceHandler:
    def __init__(self, dlq_path: str):
        self.dead_letter_queue = DeadLetterQueue(dlq_path)

    def handle_failure(self, product: ProductFeatures, error: Exception):
        """Store failed products for later retry"""
        self.dead_letter_queue.enqueue({
            'product': product.to_dict(),
            'error': str(error),
            'retry_count': 0,
            'failed_at': time.time()
        })

    def retry_failed_products(self):
        """Separate job to reprocess failures"""
        for failed_item in self.dead_letter_queue.get_retryable():
            # Retry with potentially updated model
            pass
```

### **Monitoring & Alerting**
```python
class InferenceMonitor:
    def track_metrics(self, batch_results: BatchResults):
        metrics = {
            'success_rate': batch_results.success_count / batch_results.total_count,
            'avg_inference_time': batch_results.avg_processing_time,
            'error_distribution': batch_results.error_types,
            'confidence_distribution': batch_results.confidence_histogram
        }

        # Alert if success rate drops below threshold
        if metrics['success_rate'] < 0.95:
            alert_oncall("Inference success rate below 95%", metrics)
```

---

## 3. Optimizing for Different Model Types

### **Challenge**
Different ML models have vastly different resource requirements and optimization strategies.

### **Model Type Strategies**

#### **A. Lightweight Models (< 100MB)**
```python
class LightweightModelStrategy:
    """For sklearn, linear models, small neural networks"""

    def optimize(self):
        return {
            'batch_size': 1000,  # Large batches for efficiency
            'workers_per_node': 8,  # High parallelism
            'memory_per_worker': '512MB',
            'model_loading': 'local_copy',  # Copy to each worker
            'caching_strategy': 'in_memory'
        }
```

#### **B. Heavy Models (1GB+ - Deep Learning)**
```python
class HeavyModelStrategy:
    """For large transformers, computer vision models"""

    def optimize(self):
        return {
            'batch_size': 32,  # GPU memory constraints
            'workers_per_node': 1,  # One model per GPU
            'memory_per_worker': '8GB',
            'model_loading': 'shared_memory',  # Share between processes
            'caching_strategy': 'disk_backed',
            'hardware': 'gpu_required'
        }
```

#### **C. Dynamic Model Selection**
```python
class ModelOptimizer:
    def __init__(self):
        self.strategies = {
            'lightweight': LightweightModelStrategy(),
            'medium': MediumModelStrategy(),
            'heavy': HeavyModelStrategy()
        }

    def select_strategy(self, model_metadata: ModelMetadata) -> Strategy:
        if model_metadata.size_mb < 100:
            return self.strategies['lightweight']
        elif model_metadata.requires_gpu:
            return self.strategies['heavy']
        else:
            return self.strategies['medium']

    def configure_workers(self, model_id: str):
        metadata = self.get_model_metadata(model_id)
        strategy = self.select_strategy(metadata)
        return strategy.optimize()
```

#### **D. Model-Specific Optimizations**

**TensorFlow/PyTorch Models:**
```python
class DeepLearningOptimizer:
    def optimize_inference(self, model):
        # Model quantization for faster inference
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        # TensorRT optimization for GPU
        if torch.cuda.is_available():
            model = torch_tensorrt.compile(model)

        return model
```

**Tree-Based Models (XGBoost, Random Forest):**
```python
class TreeModelOptimizer:
    def optimize_inference(self, model):
        # Enable parallel prediction
        model.set_param('nthread', os.cpu_count())

        # Use optimized prediction methods
        return model.predict_proba  # Often faster than predict
```

---

## 4. Model Versioning

### **Challenge**
Managing multiple model versions, gradual rollouts, A/B testing, and rollbacks in production.

### **Versioning Architecture**

#### **A. Semantic Versioning for ML Models**
```python
class ModelVersion:
    def __init__(self, major: int, minor: int, patch: int,
                 training_data_hash: str, experiment_id: str):
        self.version = f"{major}.{minor}.{patch}"
        self.training_data_hash = training_data_hash  # Data lineage
        self.experiment_id = experiment_id  # Experiment tracking
        self.created_at = datetime.utcnow()

    @property
    def full_version(self) -> str:
        return f"{self.version}+{self.training_data_hash[:8]}"

# Example: recommendation_model_v2.1.3+a1b2c3d4
```

#### **B. Model Registry with Metadata**
```python
class ProductionModelRegistry:
    def __init__(self):
        self.models = {}
        self.deployment_metadata = {}

    def register_model(self, model_id: str, version: ModelVersion,
                      model_artifacts: ModelArtifacts):
        """Register new model version with full metadata"""
        registry_key = f"{model_id}:{version.full_version}"

        self.models[registry_key] = {
            'model_binary': model_artifacts.model_file,
            'preprocessing': model_artifacts.preprocessor,
            'schema': model_artifacts.input_schema,
            'performance_metrics': model_artifacts.validation_metrics,
            'hardware_requirements': model_artifacts.hw_specs,
            'training_metadata': {
                'dataset_version': version.training_data_hash,
                'experiment_id': version.experiment_id,
                'training_time': model_artifacts.training_duration,
                'hyperparameters': model_artifacts.hyperparams
            }
        }

    def get_production_version(self, model_id: str) -> str:
        """Get currently deployed production version"""
        return self.deployment_metadata[model_id]['production_version']

    def promote_to_production(self, model_id: str, version: str,
                            rollout_strategy: RolloutStrategy):
        """Promote model version to production with rollout plan"""
        self.deployment_metadata[model_id] = {
            'production_version': version,
            'previous_version': self.get_production_version(model_id),
            'rollout_strategy': rollout_strategy,
            'deployed_at': datetime.utcnow()
        }
```

#### **C. Gradual Rollout Strategies**
```python
class RolloutStrategy:
    """Manage gradual model deployment"""

    def canary_deployment(self, new_version: str, traffic_percent: float = 5.0):
        """Route small percentage of traffic to new model"""
        return {
            'strategy': 'canary',
            'new_version': new_version,
            'traffic_split': traffic_percent,
            'monitoring_duration': '24h',
            'rollback_triggers': [
                'error_rate > 0.01',
                'latency_p95 > 500ms',
                'accuracy_drop > 0.05'
            ]
        }

    def blue_green_deployment(self, new_version: str):
        """Parallel deployment with instant switch"""
        return {
            'strategy': 'blue_green',
            'new_version': new_version,
            'switch_strategy': 'instant',
            'rollback_time': '< 30s'
        }

    def feature_flag_rollout(self, new_version: str, user_segments: List[str]):
        """Feature flag based deployment"""
        return {
            'strategy': 'feature_flag',
            'new_version': new_version,
            'enabled_segments': user_segments,
            'ramp_schedule': 'weekly_10_percent_increase'
        }
```

#### **D. A/B Testing Framework**
```python
class ModelABTester:
    def __init__(self, experiment_tracker: ExperimentTracker):
        self.tracker = experiment_tracker

    def run_ab_test(self, model_a: str, model_b: str,
                   test_config: ABTestConfig):
        """Compare two model versions with statistical significance"""

        def route_inference(product_id: str) -> str:
            # Deterministic routing based on product ID
            hash_val = hash(product_id) % 100
            if hash_val < test_config.traffic_split:
                return model_a
            else:
                return model_b

        # Track metrics for both models
        for product in test_products:
            model_version = route_inference(product.product_id)
            result = self.inference_engine.predict(product, model_version)

            self.tracker.log_result(
                experiment_id=test_config.experiment_id,
                model_version=model_version,
                product_id=product.product_id,
                prediction=result,
                ground_truth=product.label  # If available
            )

    def analyze_results(self, experiment_id: str) -> ABTestResults:
        """Statistical analysis of A/B test results"""
        results = self.tracker.get_experiment_results(experiment_id)

        return ABTestResults(
            statistical_significance=self._calculate_significance(results),
            performance_metrics=self._calculate_metrics(results),
            recommendation=self._make_recommendation(results)
        )
```

#### **E. Rollback Mechanisms**
```python
class ModelRollbackManager:
    def __init__(self, registry: ProductionModelRegistry):
        self.registry = registry
        self.monitoring = ModelMonitor()

    def setup_automatic_rollback(self, model_id: str, triggers: List[str]):
        """Configure automatic rollback conditions"""
        for trigger in triggers:
            self.monitoring.add_alert(
                condition=trigger,
                action=lambda: self.emergency_rollback(model_id),
                cooldown_period='5m'
            )

    def emergency_rollback(self, model_id: str):
        """Immediate rollback to previous stable version"""
        current_deployment = self.registry.deployment_metadata[model_id]
        previous_version = current_deployment['previous_version']

        # Instant traffic switch
        self.update_load_balancer_config(model_id, previous_version)

        # Alert engineering team
        self.alert_oncall(f"Emergency rollback for {model_id} to {previous_version}")

        # Update registry
        self.registry.deployment_metadata[model_id]['emergency_rollback'] = {
            'rolled_back_at': datetime.utcnow(),
            'rolled_back_from': current_deployment['production_version'],
            'rolled_back_to': previous_version,
            'trigger': 'automatic_monitoring'
        }
```

---

## Production Implementation Recommendations

### **1. Start Simple, Scale Gradually**
- Begin with model broadcasting for proof of concept
- Evolve to model registry as scale increases
- Add A/B testing framework when business metrics are established

### **2. Monitoring is Critical**
- Track inference latency, success rates, and confidence distributions
- Monitor model drift and data quality issues
- Set up automated alerts for performance degradation

### **3. Design for Failure**
- Implement circuit breakers for model services
- Have fallback models or default predictions
- Design graceful degradation strategies

### **4. Version Everything**
- Model binaries, preprocessing code, training data
- Use immutable artifacts and reproducible builds
- Maintain audit trails for compliance

This architecture provides a production-ready foundation for distributed ML inference that can scale from thousands to billions of products while maintaining reliability and performance.