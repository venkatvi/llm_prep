"""
Copyright (c) 2025. All rights reserved.
"""

"""
Calibration dataset processing for quantization.
Handles collection and processing of activation statistics.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
import numpy as np
from collections import defaultdict, OrderedDict
import random
from dataclasses import dataclass


@dataclass
class CalibrationSample:
    """A single calibration sample with metadata."""
    data: torch.Tensor
    metadata: Dict[str, Any]
    sample_id: int


class CalibrationDataset(Dataset):
    """Dataset wrapper for calibration data with advanced sampling strategies."""
    
    def __init__(
        self, 
        dataset: Dataset,
        sampling_strategy: str = "random",
        max_samples: Optional[int] = None,
        stratify_by: Optional[str] = None,
        seed: int = 42
    ):
        """Initialize calibration dataset.
        
        Args:
            dataset: Original dataset to sample from
            sampling_strategy: Strategy for sampling ("random", "diverse", "representative")
            max_samples: Maximum number of samples to use
            stratify_by: Key to stratify sampling by (e.g., "label", "length")
            seed: Random seed for reproducible sampling
        """
        self.dataset = dataset
        self.sampling_strategy = sampling_strategy
        self.max_samples = max_samples or len(dataset)
        self.stratify_by = stratify_by
        self.seed = seed
        
        # Generate sample indices
        self.sample_indices = self._generate_sample_indices()
        
    def _generate_sample_indices(self) -> List[int]:
        """Generate indices for calibration samples based on strategy."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        total_samples = len(self.dataset)
        target_samples = min(self.max_samples, total_samples)
        
        if self.sampling_strategy == "random":
            return random.sample(range(total_samples), target_samples)
            
        elif self.sampling_strategy == "diverse":
            # Try to sample diverse examples (simplified heuristic)
            indices = list(range(total_samples))
            random.shuffle(indices)
            
            # If we have a way to measure diversity, we could be more sophisticated
            return indices[:target_samples]
            
        elif self.sampling_strategy == "representative":
            # Stratified sampling if stratify_by is provided
            if self.stratify_by is not None:
                return self._stratified_sampling(target_samples)
            else:
                # Fall back to uniform sampling
                step = total_samples // target_samples
                return list(range(0, total_samples, step))[:target_samples]
                
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
    def _stratified_sampling(self, target_samples: int) -> List[int]:
        """Perform stratified sampling based on stratify_by key."""
        # This would require access to labels/metadata
        # For now, fall back to random sampling
        return random.sample(range(len(self.dataset)), target_samples)
        
    def __len__(self) -> int:
        return len(self.sample_indices)
        
    def __getitem__(self, idx: int) -> Any:
        original_idx = self.sample_indices[idx]
        return self.dataset[original_idx]


class StatisticsCollector:
    """Collects and manages activation statistics during calibration."""
    
    def __init__(self, percentiles: List[float] = [1, 5, 25, 50, 75, 95, 99]):
        """Initialize statistics collector.
        
        Args:
            percentiles: Percentiles to compute for activation distributions
        """
        self.percentiles = percentiles
        self.layer_stats = defaultdict(lambda: {
            'values': [],
            'count': 0,
            'sum': 0.0,
            'sum_sq': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'finalized': False
        })
        
    def collect_tensor_stats(self, layer_name: str, tensor: torch.Tensor) -> None:
        """Collect statistics for a tensor.
        
        Args:
            layer_name: Name of the layer producing this tensor
            tensor: Tensor to collect statistics from
        """
        if tensor.numel() == 0:
            return
            
        stats = self.layer_stats[layer_name]
        
        # Convert to CPU and flatten for statistics
        flat_tensor = tensor.detach().cpu().flatten().float()
        
        # Update running statistics
        stats['count'] += flat_tensor.numel()
        stats['sum'] += flat_tensor.sum().item()
        stats['sum_sq'] += (flat_tensor ** 2).sum().item()
        stats['min'] = min(stats['min'], flat_tensor.min().item())
        stats['max'] = max(stats['max'], flat_tensor.max().item())
        
        # # Store sample values for percentile computation (with memory limit)
        # if len(stats['values']) < 10000:  # Limit memory usage
        #     sample_size = min(1000, flat_tensor.numel())
        #     if flat_tensor.numel() > sample_size:
        #         # Sample random subset
        #         indices = torch.randperm(flat_tensor.numel())[:sample_size]
        #         sampled_values = flat_tensor[indices]
        #     else:
        #         sampled_values = flat_tensor
                
        #     stats['values'].extend(sampled_values.tolist())
            
    def finalize_stats(self) -> None:
        """Finalize statistics computation for all layers."""
        for layer_name, stats in self.layer_stats.items():
            if stats['finalized']:
                continue
                
            count = stats['count']
            if count == 0:
                continue
                
            # Compute mean and std
            mean = stats['sum'] / count
            variance = (stats['sum_sq'] / count) - (mean ** 2)
            std = np.sqrt(max(variance, 0))
            
            # Compute percentiles
            if stats['values']:
                values = np.array(stats['values'])
                percentile_values = {}
                for p in self.percentiles:
                    percentile_values[p] = np.percentile(values, p)
                stats['percentiles'] = percentile_values
                
            # Store computed statistics
            stats['mean'] = mean
            stats['std'] = std
            stats['variance'] = variance
            stats['finalized'] = True
            
            # Clear raw values to save memory
            stats['values'] = None
            
    def get_layer_stats(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Statistics dictionary or None if layer not found
        """
        return self.layer_stats.get(layer_name, None)
        
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all layers.
        
        Returns:
            Dictionary mapping layer names to statistics
        """
        return dict(self.layer_stats)
        
    def print_stats_summary(self) -> None:
        """Print summary of collected statistics."""
        print(f"📊 Statistics Summary for {len(self.layer_stats)} layers:")
        print("-" * 60)
        
        for layer_name, stats in self.layer_stats.items():
            if not stats['finalized']:
                continue
                
            print(f"Layer: {layer_name}")
            print(f"  Samples: {stats['count']:,}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            
            if 'percentiles' in stats and stats['percentiles']:
                p1, p99 = stats['percentiles'].get(1, 0), stats['percentiles'].get(99, 0)
                print(f"  P1-P99: [{p1:.4f}, {p99:.4f}]")
            print()
            

class CalibrationManager:
    """Manages the calibration process for quantization."""
    
    def __init__(self):
        """Initialize calibration manager."""
        self.calibration_history = []
        self.current_calibration = None
        
    def create_calibration_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        sampling_strategy: str = "representative",
        max_samples: int = 512,
        num_workers: int = 0,
        seed: int = 42
    ) -> DataLoader:
        """Create optimized dataloader for calibration.
        
        Args:
            dataset: Original dataset
            batch_size: Batch size for calibration
            sampling_strategy: Strategy for sampling calibration data
            max_samples: Maximum number of samples to use
            num_workers: Number of data loading workers
            seed: Random seed for reproducibility
            
        Returns:
            DataLoader for calibration
        """
        calibration_dataset = CalibrationDataset(
            dataset=dataset,
            sampling_strategy=sampling_strategy,
            max_samples=max_samples,
            seed=seed
        )
        
        return DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            shuffle=False,  # Order doesn't matter for calibration
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
    def calibrate_model_advanced(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        forward_fn: Optional[callable] = None,
        layer_filter: Optional[callable] = None,
        progress_callback: Optional[callable] = None
    ) -> StatisticsCollector:
        """Advanced calibration with hooks and monitoring.
        
        Args:
            model: Model to calibrate
            calibration_loader: DataLoader with calibration data
            forward_fn: Custom forward function
            layer_filter: Function to filter which layers to calibrate
            progress_callback: Callback for progress updates
            
        Returns:
            StatisticsCollector with collected statistics
        """
        model.eval()
        collector = StatisticsCollector()
        hooks = []
        
        # Register forward hooks
        def create_hook(name: str):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    collector.collect_tensor_stats(name, output)
                elif isinstance(output, (tuple, list)):
                    for i, tensor in enumerate(output):
                        if isinstance(tensor, torch.Tensor):
                            collector.collect_tensor_stats(f"{name}_output_{i}", tensor)
            return hook_fn
            
        # Register hooks for selected layers
        for name, module in model.named_modules():
            if self._should_calibrate_layer(module, name, layer_filter):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
                
        # Run calibration
        total_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_loader):
                try:
                    if forward_fn is not None:
                        forward_fn(model, batch)
                    else:
                        # Default forward pass
                        if isinstance(batch, (tuple, list)):
                            if len(batch) >= 1:
                                model(batch[0])
                        else:
                            model(batch)
                            
                    batch_size = batch[0].size(0) if isinstance(batch, (tuple, list)) else batch.size(0)
                    total_samples += batch_size
                    
                    if progress_callback is not None:
                        progress_callback(batch_idx + 1, len(calibration_loader), total_samples)
                        
                except Exception as e:
                    print(f"Warning: Error processing batch {batch_idx}: {e}")
                    continue
                    
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Finalize statistics
        collector.finalize_stats()
        
        # Store calibration record
        self.current_calibration = {
            'total_samples': total_samples,
            'total_batches': len(calibration_loader),
            'collector': collector
        }
        self.calibration_history.append(self.current_calibration)
        
        return collector
        
    def _should_calibrate_layer(
        self, 
        module: nn.Module, 
        name: str, 
        layer_filter: Optional[callable]
    ) -> bool:
        """Determine if a layer should be calibrated."""
        # Apply custom filter if provided
        if layer_filter is not None:
            return layer_filter(module, name)
            
        # Default: calibrate Linear and Embedding layers
        return isinstance(module, (nn.Linear, nn.Embedding)) and not isinstance(module, nn.LayerNorm)
        
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration process."""
        if self.current_calibration is None:
            return {"status": "No calibration performed"}
            
        return {
            "status": "Calibration completed",
            "total_samples": self.current_calibration['total_samples'],
            "total_batches": self.current_calibration['total_batches'],
            "layers_calibrated": len(self.current_calibration['collector'].layer_stats),
            "calibration_count": len(self.calibration_history)
        }
        

class ActivationRangeEstimator:
    """Estimates optimal activation ranges for quantization."""
    
    def __init__(self, method: str = "percentile"):
        """Initialize range estimator.
        
        Args:
            method: Method for range estimation ("percentile", "minmax", "entropy", "mse")
        """
        self.method = method
        
    def estimate_range(
        self, 
        stats: Dict[str, Any], 
        target_percentile: Tuple[float, float] = (1.0, 99.0)
    ) -> Tuple[float, float]:
        """Estimate optimal quantization range from statistics.
        
        Args:
            stats: Layer statistics
            target_percentile: Target percentile range for clipping
            
        Returns:
            Tuple of (min_val, max_val) for quantization
        """
        if self.method == "percentile":
            if 'percentiles' in stats and stats['percentiles']:
                low_pct, high_pct = target_percentile
                min_val = stats['percentiles'].get(low_pct, stats['min'])
                max_val = stats['percentiles'].get(high_pct, stats['max'])
                return min_val, max_val
            else:
                return stats['min'], stats['max']
                
        elif self.method == "minmax":
            return stats['min'], stats['max']
            
        elif self.method == "entropy":
            # Use entropy-based range estimation (simplified)
            if 'mean' in stats and 'std' in stats:
                mean, std = stats['mean'], stats['std']
                # Use 3-sigma rule as approximation
                min_val = mean - 3 * std
                max_val = mean + 3 * std
                return max(min_val, stats['min']), min(max_val, stats['max'])
            else:
                return stats['min'], stats['max']
                
        elif self.method == "mse":
            # MSE-based range estimation would require iterative optimization
            # For now, fall back to percentile method
            return self.estimate_range(stats, target_percentile)
            
        else:
            raise ValueError(f"Unknown range estimation method: {self.method}")
            

def create_text_calibration_dataloader(
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 8,
    max_samples: int = 512
) -> DataLoader:
    """Create calibration dataloader for text data.
    
    Args:
        tokenizer: Tokenizer for text processing
        texts: List of text strings
        max_length: Maximum sequence length
        batch_size: Batch size
        max_samples: Maximum number of samples
        
    Returns:
        DataLoader for calibration
    """
    # Sample texts if needed
    if len(texts) > max_samples:
        texts = random.sample(texts, max_samples)
        
    # Tokenize texts
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create dataset
    class TextCalibrationDataset(Dataset):
        def __init__(self, encoded_data):
            self.input_ids = encoded_data['input_ids']
            self.attention_mask = encoded_data.get('attention_mask', None)
            
        def __len__(self):
            return len(self.input_ids)
            
        def __getitem__(self, idx):
            item = {'input_ids': self.input_ids[idx]}
            if self.attention_mask is not None:
                item['attention_mask'] = self.attention_mask[idx]
            return item
            
    dataset = TextCalibrationDataset(encoded)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    # Demo usage
    print("📊 Calibration System Demo")
    print("=" * 50)
    
    # Create mock statistics collector
    collector = StatisticsCollector()
    
    # Simulate collecting statistics
    for layer_idx in range(3):
        layer_name = f"layer_{layer_idx}"
        for _ in range(10):
            # Simulate different activation distributions
            if layer_idx == 0:
                tensor = torch.randn(32, 128) * 0.5  # Small variance
            elif layer_idx == 1:
                tensor = torch.randn(32, 128) * 2.0   # Large variance
            else:
                tensor = torch.rand(32, 128) - 0.5    # Uniform distribution
                
            collector.collect_tensor_stats(layer_name, tensor)
            
    # Finalize and print statistics
    collector.finalize_stats()
    collector.print_stats_summary()
    
    # Test range estimation
    range_estimator = ActivationRangeEstimator("percentile")
    for layer_name in ["layer_0", "layer_1", "layer_2"]:
        stats = collector.get_layer_stats(layer_name)
        if stats:
            min_val, max_val = range_estimator.estimate_range(stats, (1, 99))
            print(f"Range for {layer_name}: [{min_val:.4f}, {max_val:.4f}]")
            
    print("\n✅ Calibration system ready!")