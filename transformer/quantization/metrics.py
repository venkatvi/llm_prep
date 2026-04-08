"""
Copyright (c) 2025. All rights reserved.
"""

"""
Quantization evaluation metrics and accuracy assessment tools.

This module provides comprehensive evaluation capabilities for quantized models
including accuracy metrics, perplexity evaluation, and model comparison tools.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass
import math


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    loss: Optional[float] = None
    latency_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None


@dataclass
class ComparisonResult:
    """Results from comparing two models."""
    original_result: EvaluationResult
    quantized_result: EvaluationResult
    accuracy_degradation: Optional[float] = None
    perplexity_increase: Optional[float] = None
    speedup: Optional[float] = None
    memory_reduction: Optional[float] = None
    compression_ratio: Optional[float] = None


class QuantizationMetrics:
    """Comprehensive quantization metrics collection and analysis."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics_history = []
        self.layer_metrics = defaultdict(dict)
        
    def compute_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Compute model size in different units."""
        total_params = 0
        total_bytes = 0
        
        for param in model.parameters():
            total_params += param.numel()
            total_bytes += param.numel() * param.element_size()
            
        for buffer in model.buffers():
            total_bytes += buffer.numel() * buffer.element_size()
            
        return {
            'total_parameters': total_params,
            'size_bytes': total_bytes,
            'size_kb': total_bytes / 1024,
            'size_mb': total_bytes / (1024 * 1024),
            'size_gb': total_bytes / (1024 * 1024 * 1024)
        }
        
    def compute_compression_ratio(
        self, 
        original_model: nn.Module, 
        quantized_model: nn.Module
    ) -> Dict[str, float]:
        """Compute compression metrics between original and quantized models."""
        orig_size = self.compute_model_size(original_model)
        quant_size = self.compute_model_size(quantized_model)
        
        return {
            'parameter_compression': orig_size['total_parameters'] / quant_size['total_parameters'],
            'size_compression': orig_size['size_bytes'] / quant_size['size_bytes'],
            'original_size_mb': orig_size['size_mb'],
            'quantized_size_mb': quant_size['size_mb'],
            'size_reduction_mb': orig_size['size_mb'] - quant_size['size_mb'],
            'size_reduction_percentage': (1 - quant_size['size_bytes'] / orig_size['size_bytes']) * 100
        }
        
    def measure_inference_latency(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Measure model inference latency."""
        model.eval()
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_data)
                
        # Synchronize GPU if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                
        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
        
    def compute_layer_wise_error(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        input_data: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Compute layer-wise quantization error."""
        layer_errors = {}
        
        # Hook to collect activations
        original_activations = {}
        quantized_activations = {}
        
        def make_hook(name, activation_dict):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_dict[name] = output.detach()
                elif isinstance(output, (tuple, list)):
                    activation_dict[name] = [o.detach() if isinstance(o, torch.Tensor) else o for o in output]
            return hook
            
        # Register hooks
        original_hooks = []
        quantized_hooks = []
        
        for name, module in original_model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(make_hook(name, original_activations))
                original_hooks.append(hook)
                
        for name, module in quantized_model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(make_hook(name, quantized_activations))
                quantized_hooks.append(hook)
                
        # Forward pass
        with torch.no_grad():
            original_model(input_data)
            quantized_model(input_data)
            
        # Compute errors
        for layer_name in original_activations:
            if layer_name in quantized_activations:
                orig_act = original_activations[layer_name]
                quant_act = quantized_activations[layer_name]
                
                if isinstance(orig_act, torch.Tensor) and isinstance(quant_act, torch.Tensor):
                    mse = F.mse_loss(orig_act, quant_act).item()
                    mae = F.l1_loss(orig_act, quant_act).item()
                    
                    # Relative error
                    orig_norm = orig_act.norm().item()
                    if orig_norm > 0:
                        rel_error = (orig_act - quant_act).norm().item() / orig_norm
                    else:
                        rel_error = 0.0
                        
                    layer_errors[layer_name] = {
                        'mse': mse,
                        'mae': mae,
                        'relative_error': rel_error
                    }
                    
        # Clean up hooks
        for hook in original_hooks + quantized_hooks:
            hook.remove()
            
        return layer_errors
        
    def analyze_quantization_sensitivity(
        self,
        model: nn.Module,
        layer_errors: Dict[str, Dict[str, float]],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Analyze which layers are most sensitive to quantization."""
        
        # Sort layers by error
        layer_sensitivity = []
        for layer_name, errors in layer_errors.items():
            avg_error = (errors['mse'] + errors['mae'] + errors['relative_error']) / 3
            layer_sensitivity.append((layer_name, avg_error))
            
        layer_sensitivity.sort(key=lambda x: x[1], reverse=True)
        
        # Identify sensitive layers
        sensitive_layers = [name for name, error in layer_sensitivity if error > threshold]
        
        return {
            'layer_sensitivity_ranking': layer_sensitivity,
            'sensitive_layers': sensitive_layers,
            'most_sensitive_layer': layer_sensitivity[0] if layer_sensitivity else None,
            'least_sensitive_layer': layer_sensitivity[-1] if layer_sensitivity else None,
            'avg_sensitivity': np.mean([error for _, error in layer_sensitivity]) if layer_sensitivity else 0,
            'sensitivity_threshold': threshold
        }


class AccuracyEvaluator:
    """Evaluates model accuracy on classification and other tasks."""
    
    def __init__(self, task_type: str = "classification"):
        """Initialize accuracy evaluator.
        
        Args:
            task_type: Type of task ("classification", "regression", "generation")
        """
        self.task_type = task_type
        
    def evaluate_classification(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cpu",
        top_k: List[int] = [1, 5]
    ) -> Dict[str, float]:
        """Evaluate classification accuracy."""
        model.eval()
        model = model.to(device)
        
        total_samples = 0
        correct_predictions = {k: 0 for k in top_k}
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                batch_size = targets.size(0)
                total_samples += batch_size
                
                # Top-k accuracy
                _, predicted = outputs.topk(max(top_k), 1, True, True)
                predicted = predicted.t()
                correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
                
                for k in top_k:
                    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    correct_predictions[k] += correct_k.item()
                    
        # Calculate metrics
        results = {
            'loss': total_loss / len(dataloader),
            'total_samples': total_samples
        }
        
        for k in top_k:
            results[f'top_{k}_accuracy'] = correct_predictions[k] / total_samples
            
        return results
        
    def evaluate_regression(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Evaluate regression metrics."""
        model.eval()
        model = model.to(device)
        
        total_samples = 0
        total_mse = 0.0
        total_mae = 0.0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                batch_size = targets.size(0)
                total_samples += batch_size
                
                mse = F.mse_loss(outputs, targets, reduction='sum')
                mae = F.l1_loss(outputs, targets, reduction='sum')
                
                total_mse += mse.item()
                total_mae += mae.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
                
        # Calculate metrics
        mse = total_mse / total_samples
        mae = total_mae / total_samples
        rmse = math.sqrt(mse)
        
        # R² score
        predictions = np.array(predictions)
        targets_array = np.array(targets_list)
        ss_res = np.sum((targets_array - predictions) ** 2)
        ss_tot = np.sum((targets_array - np.mean(targets_array)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mse': mse,
            'mae': mae, 
            'rmse': rmse,
            'r2_score': r2_score,
            'total_samples': total_samples
        }


class PerplexityEvaluator:
    """Evaluates language model perplexity."""
    
    def __init__(self):
        """Initialize perplexity evaluator."""
        pass
        
    def evaluate_perplexity(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cpu",
        max_length: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model perplexity on language modeling task."""
        model.eval()
        model = model.to(device)
        
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch.get('attention_mask')
                    labels = batch.get('labels', input_ids)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_ids = batch[0].to(device)
                    labels = batch[1].to(device)
                    attention_mask = None
                else:
                    input_ids = batch.to(device)
                    labels = input_ids
                    attention_mask = None
                    
                # Limit sequence length if specified
                if max_length is not None:
                    input_ids = input_ids[:, :max_length]
                    labels = labels[:, :max_length]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :max_length]
                        
                # Forward pass
                if attention_mask is not None:
                    outputs = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids)
                    
                # Get logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[0]
                    
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten tensors
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Calculate loss
                loss = criterion(shift_logits, shift_labels)
                
                # Count valid tokens
                valid_tokens = (shift_labels != -100).sum().item()
                
                total_loss += loss.item()
                total_tokens += valid_tokens
                
        # Calculate perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')  # Avoid overflow
        
        return {
            'perplexity': perplexity,
            'loss': avg_loss,
            'total_tokens': total_tokens,
            'total_batches': len(dataloader)
        }
        
    def evaluate_generation_quality(
        self,
        model: nn.Module,
        tokenizer,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Evaluate generation quality metrics."""
        model.eval()
        model = model.to(device)
        
        generated_texts = []
        generation_lengths = []
        generation_times = []
        
        for prompt in prompts:
            # Encode prompt
            inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # Generate with timing
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            end_time = time.time()
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            
            generation_lengths.append(len(outputs[0]))
            generation_times.append(end_time - start_time)
            
        return {
            'generated_texts': generated_texts,
            'avg_generation_length': np.mean(generation_lengths),
            'avg_generation_time': np.mean(generation_times),
            'total_prompts': len(prompts)
        }


class ModelComparator:
    """Compares original and quantized models comprehensively."""
    
    def __init__(self):
        """Initialize model comparator."""
        self.metrics = QuantizationMetrics()
        self.accuracy_evaluator = AccuracyEvaluator()
        self.perplexity_evaluator = PerplexityEvaluator()
        
    def comprehensive_comparison(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        eval_dataloader: DataLoader,
        sample_input: torch.Tensor,
        task_type: str = "classification",
        device: str = "cpu"
    ) -> ComparisonResult:
        """Perform comprehensive comparison between models."""
        
        # Move models to device
        original_model = original_model.to(device)
        quantized_model = quantized_model.to(device)
        sample_input = sample_input.to(device)
        
        # Evaluate original model
        print("Evaluating original model...")
        original_result = self._evaluate_single_model(
            original_model, eval_dataloader, sample_input, task_type, device
        )
        
        # Evaluate quantized model
        print("Evaluating quantized model...")
        quantized_result = self._evaluate_single_model(
            quantized_model, eval_dataloader, sample_input, task_type, device
        )
        
        # Compute comparison metrics
        comparison = ComparisonResult(
            original_result=original_result,
            quantized_result=quantized_result
        )
        
        # Accuracy degradation
        if original_result.accuracy and quantized_result.accuracy:
            comparison.accuracy_degradation = original_result.accuracy - quantized_result.accuracy
            
        # Perplexity increase
        if original_result.perplexity and quantized_result.perplexity:
            comparison.perplexity_increase = quantized_result.perplexity - original_result.perplexity
            
        # Speedup
        if original_result.latency_ms and quantized_result.latency_ms:
            comparison.speedup = original_result.latency_ms / quantized_result.latency_ms
            
        # Memory reduction
        if original_result.memory_mb and quantized_result.memory_mb:
            comparison.memory_reduction = original_result.memory_mb - quantized_result.memory_mb
            
        # Compression ratio
        compression_metrics = self.metrics.compute_compression_ratio(original_model, quantized_model)
        comparison.compression_ratio = compression_metrics['size_compression']
        
        return comparison
        
    def _evaluate_single_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        sample_input: torch.Tensor,
        task_type: str,
        device: str
    ) -> EvaluationResult:
        """Evaluate a single model."""
        result = EvaluationResult()
        
        # Model size
        size_info = self.metrics.compute_model_size(model)
        result.memory_mb = size_info['size_mb']
        
        # Latency measurement
        latency_info = self.metrics.measure_inference_latency(model, sample_input)
        result.latency_ms = latency_info['mean_latency_ms']
        
        # Task-specific evaluation
        if task_type == "classification":
            eval_results = self.accuracy_evaluator.evaluate_classification(model, dataloader, device)
            result.accuracy = eval_results.get('top_1_accuracy')
            result.loss = eval_results.get('loss')
            
        elif task_type == "language_modeling":
            eval_results = self.perplexity_evaluator.evaluate_perplexity(model, dataloader, device)
            result.perplexity = eval_results.get('perplexity')
            result.loss = eval_results.get('loss')
            
        elif task_type == "regression":
            eval_results = self.accuracy_evaluator.evaluate_regression(model, dataloader, device)
            result.loss = eval_results.get('mse')
            result.additional_metrics = {
                'mae': eval_results.get('mae'),
                'rmse': eval_results.get('rmse'),
                'r2_score': eval_results.get('r2_score')
            }
            
        return result
        
    def print_comparison_report(self, comparison: ComparisonResult):
        """Print detailed comparison report."""
        print("\n" + "="*60)
        print("📊 QUANTIZATION COMPARISON REPORT")
        print("="*60)
        
        # Model sizes
        orig_mem = comparison.original_result.memory_mb or 0
        quant_mem = comparison.quantized_result.memory_mb or 0
        print(f"\n📏 Model Size:")
        print(f"  Original:   {orig_mem:.2f} MB")
        print(f"  Quantized:  {quant_mem:.2f} MB")
        print(f"  Reduction:  {comparison.memory_reduction:.2f} MB ({100*comparison.memory_reduction/orig_mem:.1f}%)")
        print(f"  Compression: {comparison.compression_ratio:.2f}x")
        
        # Performance
        orig_lat = comparison.original_result.latency_ms or 0
        quant_lat = comparison.quantized_result.latency_ms or 0
        print(f"\n⚡ Performance:")
        print(f"  Original Latency:   {orig_lat:.2f} ms")
        print(f"  Quantized Latency:  {quant_lat:.2f} ms")
        print(f"  Speedup:           {comparison.speedup:.2f}x")
        
        # Accuracy
        if comparison.original_result.accuracy and comparison.quantized_result.accuracy:
            print(f"\n🎯 Accuracy:")
            print(f"  Original:    {comparison.original_result.accuracy:.4f}")
            print(f"  Quantized:   {comparison.quantized_result.accuracy:.4f}")
            print(f"  Degradation: {comparison.accuracy_degradation:.4f} ({100*comparison.accuracy_degradation/comparison.original_result.accuracy:.2f}%)")
        
        # Perplexity
        if comparison.original_result.perplexity and comparison.quantized_result.perplexity:
            print(f"\n📈 Perplexity:")
            print(f"  Original:  {comparison.original_result.perplexity:.2f}")
            print(f"  Quantized: {comparison.quantized_result.perplexity:.2f}")
            print(f"  Increase:  {comparison.perplexity_increase:.2f}")
            
        print("\n" + "="*60)


if __name__ == "__main__":
    # Demo usage
    print("📊 Quantization Metrics Demo")
    print("="*50)
    
    # Create simple models for testing
    original_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # Simulate quantized model (just smaller weights for demo)
    quantized_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(), 
        nn.Linear(64, 10)
    )
    
    # Scale weights to simulate quantization
    with torch.no_grad():
        for param in quantized_model.parameters():
            param.mul_(0.95)  # Simulate slight degradation
            
    # Test metrics
    metrics = QuantizationMetrics()
    
    # Model sizes
    orig_size = metrics.compute_model_size(original_model)
    quant_size = metrics.compute_model_size(quantized_model)
    
    print(f"Original model: {orig_size['size_kb']:.2f} KB")
    print(f"Quantized model: {quant_size['size_kb']:.2f} KB")
    
    # Compression ratio
    compression = metrics.compute_compression_ratio(original_model, quantized_model)
    print(f"Compression ratio: {compression['size_compression']:.2f}x")
    
    # Latency test
    sample_input = torch.randn(32, 128)
    latency_orig = metrics.measure_inference_latency(original_model, sample_input, num_runs=10)
    latency_quant = metrics.measure_inference_latency(quantized_model, sample_input, num_runs=10)
    
    print(f"Original latency: {latency_orig['mean_latency_ms']:.3f} ms")
    print(f"Quantized latency: {latency_quant['mean_latency_ms']:.3f} ms")
    print(f"Speedup: {latency_orig['mean_latency_ms']/latency_quant['mean_latency_ms']:.2f}x")
    
    print("\n✅ Metrics system ready!")