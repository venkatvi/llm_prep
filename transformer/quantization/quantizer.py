"""
Copyright (c) 2025. All rights reserved.
"""

"""
Main quantization framework for transformer models.
Supports INT8 and INT4 post-training quantization with calibration.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import warnings
from collections import OrderedDict

from .calibration import CalibrationManager, StatisticsCollector
from .layers import QuantizedLinear, QuantizedEmbedding, QuantizedAttention, QuantizedFFN


class QuantizationScheme(Enum):
    """Quantization schemes supported."""
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"  # Different layers use different schemes


class LayerQuantizationStrategy(Enum):
    """Strategy for quantizing different layer types."""
    UNIFORM = "uniform"  # Same quantization for all layers
    LAYER_WISE = "layer_wise"  # Different quantization per layer type
    SENSITIVITY_BASED = "sensitivity_based"  # Based on accuracy sensitivity
    MIXED_PRECISION = "mixed_precision"  # Mixed INT8/INT4 based on importance


@dataclass
class QuantizationConfig:
    """Configuration for transformer quantization.
    
    Attributes:
        scheme: Primary quantization scheme (INT8/INT4/MIXED)
        strategy: Layer quantization strategy
        calibration_samples: Number of samples for calibration
        percentile_range: Percentile range for activation clipping (e.g., (1, 99))
        weight_only: Whether to quantize only weights (not activations)
        preserve_embeddings: Whether to keep embeddings in FP32
        preserve_layer_norm: Whether to keep layer norms in FP32
        symmetric_weights: Use symmetric quantization for weights
        symmetric_activations: Use symmetric quantization for activations
        per_channel_weights: Use per-channel quantization for weights
        mixed_precision_threshold: Sensitivity threshold for mixed precision
        enable_kv_cache_quantization: Quantize KV cache tensors
        quantize_attention_weights: Quantize attention weight matrices
        quantize_ffn_weights: Quantize FFN weight matrices
        quantize_output_weights: Quantize output projection weights
    """
    scheme: QuantizationScheme = QuantizationScheme.INT8
    strategy: LayerQuantizationStrategy = LayerQuantizationStrategy.LAYER_WISE
    calibration_samples: int = 512
    percentile_range: Tuple[float, float] = (1.0, 99.0)
    weight_only: bool = False
    preserve_embeddings: bool = True
    preserve_layer_norm: bool = True
    symmetric_weights: bool = True
    symmetric_activations: bool = False
    per_channel_weights: bool = True
    mixed_precision_threshold: float = 0.05  # 5% accuracy drop threshold
    enable_kv_cache_quantization: bool = False
    quantize_attention_weights: bool = True
    quantize_ffn_weights: bool = True
    quantize_output_weights: bool = True


class TransformerQuantizer:
    """Main quantizer for transformer models with advanced quantization strategies."""
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """Initialize transformer quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self.calibration_manager = CalibrationManager()
        self.statistics_collector = StatisticsCollector()
        
        # Layer sensitivity analysis results
        self.layer_sensitivities = {}
        self.quantization_parameters = {}
        
        # Quantization statistics
        self.quantization_stats = {
            "total_parameters": 0,
            "quantized_parameters": 0,
            "compression_ratio": 0.0,
            "memory_savings": 0.0,
            "layers_quantized": 0,
            "layers_preserved": 0
        }
        
    def quantize_model(
        self, 
        model: nn.Module, 
        calibration_data: Optional[torch.utils.data.DataLoader] = None,
        forward_fn: Optional[Callable] = None
    ) -> nn.Module:
        """Quantize a transformer model with calibration.
        
        Args:
            model: Model to quantize
            calibration_data: Data loader for calibration
            forward_fn: Custom forward function for calibration
            
        Returns:
            Quantized model
        """
        model.eval()
        
        # Step 1: Collect activation statistics via calibration
        if calibration_data is not None:
            print("🔧 Collecting activation statistics...")
            self._calibrate_model(model, calibration_data, forward_fn)
        
        # Step 2: Analyze layer sensitivity (optional for mixed precision)
        if self.config.strategy == LayerQuantizationStrategy.SENSITIVITY_BASED:
            print("📊 Analyzing layer sensitivities...")
            self._analyze_layer_sensitivity(model, calibration_data, forward_fn)
            
        # Step 3: Determine quantization scheme per layer
        layer_schemes = self._determine_layer_schemes(model)
        
        # Step 4: Quantize model layers
        print("⚡ Quantizing model layers...")
        quantized_model = self._quantize_layers(model, layer_schemes)
        
        # Step 5: Update statistics
        self._update_quantization_stats(model, quantized_model)
        
        print(f"✅ Quantization complete! Compression ratio: {self.quantization_stats['compression_ratio']:.2f}x")
        return quantized_model
        
    def _calibrate_model(
        self, 
        model: nn.Module, 
        calibration_data: torch.utils.data.DataLoader,
        forward_fn: Optional[Callable] = None
    ) -> None:
        """Calibrate model to collect activation statistics."""
        # Register hooks to collect activation statistics
        hooks = []
        
        def register_hook(name: str, module: nn.Module):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.statistics_collector.collect_tensor_stats(name, output)
                elif isinstance(output, (tuple, list)):
                    for i, tensor in enumerate(output):
                        if isinstance(tensor, torch.Tensor):
                            self.statistics_collector.collect_tensor_stats(f"{name}_output_{i}", tensor)
                            
            return module.register_forward_hook(hook_fn)
            
        # Register hooks for quantizable layers
        for name, module in model.named_modules():
            if self._should_collect_stats(module):
                hooks.append(register_hook(name, module))
                
        # Run calibration
        with torch.no_grad():
            sample_count = 0
            for batch_idx, batch in enumerate(calibration_data):
                if sample_count >= self.config.calibration_samples:
                    break
                    
                if forward_fn is not None:
                    forward_fn(model, batch)
                else:
                    # Default forward pass
                    if isinstance(batch, (tuple, list)):
                        model(*batch)
                    else:
                        model(batch)
                        
                sample_count += batch[0].size(0) if isinstance(batch, (tuple, list)) else batch.size(0)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {sample_count} samples...")
                    
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Finalize statistics
        self.statistics_collector.finalize_stats()
        print(f"📈 Collected statistics from {sample_count} calibration samples")
        
    def _should_collect_stats(self, module: nn.Module) -> bool:
        """Determine if we should collect statistics for a module."""
        return isinstance(module, (nn.Linear, nn.Embedding)) and not isinstance(module, nn.LayerNorm)
        
    def _analyze_layer_sensitivity(
        self, 
        model: nn.Module, 
        calibration_data: torch.utils.data.DataLoader,
        forward_fn: Optional[Callable] = None
    ) -> None:
        """Analyze sensitivity of different layers to quantization."""
        if calibration_data is None:
            warnings.warn("Cannot perform sensitivity analysis without calibration data")
            return
            
        # Get baseline accuracy
        baseline_metrics = self._evaluate_model_subset(model, calibration_data, forward_fn)
        
        # Test quantization sensitivity for each layer type
        layer_types = ['embedding', 'attention', 'ffn', 'output']
        
        for layer_type in layer_types:
            # Create temporarily quantized model for this layer type only
            temp_model = self._create_partial_quantized_model(model, layer_type)
            
            # Evaluate accuracy
            quantized_metrics = self._evaluate_model_subset(temp_model, calibration_data, forward_fn)
            
            # Calculate sensitivity (accuracy drop)
            sensitivity = baseline_metrics - quantized_metrics
            self.layer_sensitivities[layer_type] = sensitivity
            
            print(f"  Layer type {layer_type}: sensitivity = {sensitivity:.4f}")
            
    def _evaluate_model_subset(
        self, 
        model: nn.Module, 
        data_loader: torch.utils.data.DataLoader,
        forward_fn: Optional[Callable] = None,
        max_samples: int = 100
    ) -> float:
        """Quick evaluation on subset of data for sensitivity analysis."""
        model.eval()
        total_loss = 0.0
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if sample_count >= max_samples:
                    break
                    
                if forward_fn is not None:
                    loss = forward_fn(model, batch)
                else:
                    # Simple loss calculation
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                        outputs = model(inputs)
                        if hasattr(outputs, 'logits'):
                            outputs = outputs.logits
                        loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    else:
                        continue
                        
                total_loss += loss.item()
                sample_count += 1
                
        return total_loss / max(sample_count, 1)
        
    def _create_partial_quantized_model(self, model: nn.Module, layer_type: str) -> nn.Module:
        """Create model with only specific layer type quantized for sensitivity testing."""
        # This would create a temporary quantized version
        # For now, return the original model (placeholder)
        return model
        
    def _determine_layer_schemes(self, model: nn.Module) -> Dict[str, QuantizationScheme]:
        """Determine quantization scheme for each layer based on strategy."""
        layer_schemes = {}
        
        if self.config.strategy == LayerQuantizationStrategy.UNIFORM:
            # All layers use the same scheme
            for name, module in model.named_modules():
                if self._should_quantize_layer(name, module):
                    layer_schemes[name] = self.config.scheme
                    
        elif self.config.strategy == LayerQuantizationStrategy.LAYER_WISE:
            # Different schemes for different layer types
            for name, module in model.named_modules():
                if self._should_quantize_layer(name, module):
                    if isinstance(module, nn.Embedding):
                        # Embeddings are more sensitive, use higher precision if requested
                        layer_schemes[name] = QuantizationScheme.INT8 if not self.config.preserve_embeddings else None
                    elif 'attention' in name.lower() or 'attn' in name.lower():
                        layer_schemes[name] = QuantizationScheme.INT8  # Attention layers are sensitive
                    elif 'ffn' in name.lower() or 'mlp' in name.lower():
                        layer_schemes[name] = self.config.scheme  # FFN can handle more aggressive quantization
                    else:
                        layer_schemes[name] = self.config.scheme
                        
        elif self.config.strategy == LayerQuantizationStrategy.SENSITIVITY_BASED:
            # Use sensitivity analysis results
            for name, module in model.named_modules():
                if self._should_quantize_layer(name, module):
                    layer_type = self._get_layer_type(name, module)
                    sensitivity = self.layer_sensitivities.get(layer_type, 0.0)
                    
                    if sensitivity > self.config.mixed_precision_threshold:
                        layer_schemes[name] = QuantizationScheme.INT8  # High sensitivity -> higher precision
                    else:
                        layer_schemes[name] = self.config.scheme  # Low sensitivity -> can use lower precision
                        
        elif self.config.strategy == LayerQuantizationStrategy.MIXED_PRECISION:
            # Strategic mixed precision based on layer importance
            for name, module in model.named_modules():
                if self._should_quantize_layer(name, module):
                    if any(critical in name.lower() for critical in ['embed', 'output', 'lm_head']):
                        layer_schemes[name] = QuantizationScheme.INT8  # Critical layers
                    elif 'attention' in name.lower():
                        layer_schemes[name] = QuantizationScheme.INT8  # Attention layers
                    else:
                        layer_schemes[name] = QuantizationScheme.INT4  # Other layers
                        
        return layer_schemes
        
    def _should_quantize_layer(self, name: str, module: nn.Module) -> bool:
        """Determine if a layer should be quantized."""
        # Skip layer norm and similar normalization layers
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            return not self.config.preserve_layer_norm
            
        # Skip embeddings if configured to preserve
        if isinstance(module, nn.Embedding):
            return not self.config.preserve_embeddings
            
        # Quantize linear layers based on configuration
        if isinstance(module, nn.Linear):
            if 'attention' in name.lower() or 'attn' in name.lower():
                return self.config.quantize_attention_weights
            elif 'ffn' in name.lower() or 'mlp' in name.lower():
                return self.config.quantize_ffn_weights
            elif 'output' in name.lower() or 'lm_head' in name.lower():
                return self.config.quantize_output_weights
            else:
                return True  # Default: quantize other linear layers
                
        return False
        
    def _get_layer_type(self, name: str, module: nn.Module) -> str:
        """Get layer type for sensitivity analysis."""
        if isinstance(module, nn.Embedding):
            return 'embedding'
        elif 'attention' in name.lower() or 'attn' in name.lower():
            return 'attention'
        elif 'ffn' in name.lower() or 'mlp' in name.lower():
            return 'ffn'
        elif 'output' in name.lower() or 'lm_head' in name.lower():
            return 'output'
        else:
            return 'other'
            
    def _quantize_layers(self, model: nn.Module, layer_schemes: Dict[str, QuantizationScheme]) -> nn.Module:
        """Replace model layers with quantized versions."""
        quantized_model = model
        
        # Collect quantization parameters for each layer
        quantization_params = {}
        
        for name, scheme in layer_schemes.items():
            if scheme is None:
                continue
                
            # Get module
            module = dict(model.named_modules())[name]
            
            # Get activation statistics
            stats = self.statistics_collector.get_layer_stats(name)
            
            # Calculate quantization parameters
            if isinstance(module, nn.Linear):
                params = self._calculate_linear_quantization_params(module, stats, scheme)
            elif isinstance(module, nn.Embedding):
                params = self._calculate_embedding_quantization_params(module, stats, scheme)
            else:
                continue
                
            quantization_params[name] = params
            
        # Replace layers with quantized versions
        quantized_model = self._replace_layers_with_quantized(model, layer_schemes, quantization_params)
        
        return quantized_model
        
    def _calculate_linear_quantization_params(
        self, 
        module: nn.Linear, 
        stats: Optional[Dict], 
        scheme: QuantizationScheme
    ) -> Dict[str, Any]:
        """Calculate quantization parameters for linear layer."""
        params = {}
        
        # Weight quantization parameters
        weight = module.weight.data
        if scheme == QuantizationScheme.INT8:
            qmin, qmax = -128, 127
        elif scheme == QuantizationScheme.INT4:
            qmin, qmax = -8, 7
        else:
            qmin, qmax = -128, 127
            
        if self.config.per_channel_weights:
            # Per-channel quantization
            if self.config.symmetric_weights:
                weight_scale = weight.abs().max(dim=1, keepdim=True)[0] / (qmax - qmin // 2)
                weight_zero_point = torch.zeros_like(weight_scale, dtype=torch.int8)
            else:
                weight_min = weight.min(dim=1, keepdim=True)[0]
                weight_max = weight.max(dim=1, keepdim=True)[0]
                weight_scale = (weight_max - weight_min) / (qmax - qmin)
                weight_zero_point = qmin - (weight_min / weight_scale).round()
        else:
            # Per-tensor quantization
            if self.config.symmetric_weights:
                weight_scale = weight.abs().max() / (qmax - qmin // 2)
                weight_zero_point = torch.tensor(0, dtype=torch.int8)
            else:
                weight_min, weight_max = weight.min(), weight.max()
                weight_scale = (weight_max - weight_min) / (qmax - qmin)
                weight_zero_point = qmin - (weight_min / weight_scale).round()
                
        params['weight_scale'] = weight_scale
        params['weight_zero_point'] = weight_zero_point
        params['weight_qmin'] = qmin
        params['weight_qmax'] = qmax
        
        # Activation quantization parameters (if available from calibration)
        if stats is not None and not self.config.weight_only:
            act_min, act_max = self._get_activation_range(stats)
            if self.config.symmetric_activations:
                act_scale = max(abs(act_min), abs(act_max)) / (qmax - qmin // 2)
                act_zero_point = torch.tensor(0, dtype=torch.int8)
            else:
                act_scale = (act_max - act_min) / (qmax - qmin)
                act_zero_point = qmin - (act_min / act_scale).round()
                
            params['activation_scale'] = act_scale
            params['activation_zero_point'] = act_zero_point
            params['activation_qmin'] = qmin
            params['activation_qmax'] = qmax
            
        return params
        
    def _calculate_embedding_quantization_params(
        self, 
        module: nn.Embedding, 
        stats: Optional[Dict], 
        scheme: QuantizationScheme
    ) -> Dict[str, Any]:
        """Calculate quantization parameters for embedding layer."""
        params = {}
        
        weight = module.weight.data
        if scheme == QuantizationScheme.INT8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = -8, 7
            
        # Embeddings typically use symmetric quantization
        weight_scale = weight.abs().max() / (qmax - qmin // 2)
        weight_zero_point = torch.tensor(0, dtype=torch.int8)
        
        params['weight_scale'] = weight_scale
        params['weight_zero_point'] = weight_zero_point
        params['weight_qmin'] = qmin
        params['weight_qmax'] = qmax
        
        return params
        
    def _get_activation_range(self, stats: Dict) -> Tuple[float, float]:
        """Get activation range from collected statistics."""
        if 'percentiles' in stats:
            low_pct, high_pct = self.config.percentile_range
            act_min = stats['percentiles'].get(low_pct, stats.get('min', 0.0))
            act_max = stats['percentiles'].get(high_pct, stats.get('max', 1.0))
        else:
            act_min = stats.get('min', 0.0)
            act_max = stats.get('max', 1.0)
            
        return float(act_min), float(act_max)
        
    def _replace_layers_with_quantized(
        self, 
        model: nn.Module, 
        layer_schemes: Dict[str, QuantizationScheme],
        quantization_params: Dict[str, Dict[str, Any]]
    ) -> nn.Module:
        """Replace original layers with quantized versions."""
        # Create a copy of the model
        quantized_model = type(model)(model.config if hasattr(model, 'config') else None)
        quantized_model.load_state_dict(model.state_dict())
        
        # Replace layers
        for name, scheme in layer_schemes.items():
            if scheme is None or name not in quantization_params:
                continue
                
            # Navigate to parent module
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = dict(quantized_model.named_modules())[parent_name]
            else:
                parent_module = quantized_model
                
            original_module = getattr(parent_module, attr_name)
            params = quantization_params[name]
            
            # Create quantized replacement
            if isinstance(original_module, nn.Linear):
                quantized_layer = QuantizedLinear.from_module(original_module, params, scheme)
            elif isinstance(original_module, nn.Embedding):
                quantized_layer = QuantizedEmbedding.from_module(original_module, params, scheme)
            else:
                continue
                
            # Replace the layer
            setattr(parent_module, attr_name, quantized_layer)
            self.quantization_stats["layers_quantized"] += 1
            
        return quantized_model
        
    def _update_quantization_stats(self, original_model: nn.Module, quantized_model: nn.Module) -> None:
        """Update quantization statistics."""
        original_params = sum(p.numel() for p in original_model.parameters())
        quantized_params = sum(p.numel() for p in quantized_model.parameters())
        
        # Calculate compression ratio (this is simplified)
        original_size = sum(p.numel() * 4 for p in original_model.parameters())  # Assume FP32
        quantized_size = 0
        
        for name, module in quantized_model.named_modules():
            if hasattr(module, 'get_memory_usage'):
                quantized_size += module.get_memory_usage()
            elif hasattr(module, 'weight'):
                quantized_size += module.weight.numel() * 4  # Still FP32 if not quantized
                
        self.quantization_stats.update({
            "total_parameters": original_params,
            "quantized_parameters": quantized_params,
            "compression_ratio": original_size / max(quantized_size, 1),
            "memory_savings": (original_size - quantized_size) / original_size * 100
        })
        
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get comprehensive quantization statistics."""
        stats = self.quantization_stats.copy()
        stats.update({
            "layer_sensitivities": self.layer_sensitivities,
            "config": {
                "scheme": self.config.scheme.value,
                "strategy": self.config.strategy.value,
                "calibration_samples": self.config.calibration_samples,
                "weight_only": self.config.weight_only
            }
        })
        return stats
        
    def save_quantized_model(self, model: nn.Module, path: str) -> None:
        """Save quantized model to disk."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'quantization_config': self.config,
            'quantization_stats': self.quantization_stats,
            'quantization_parameters': self.quantization_parameters
        }, path)
        print(f"💾 Quantized model saved to {path}")
        
    def load_quantized_model(self, model: nn.Module, path: str) -> nn.Module:
        """Load quantized model from disk."""
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['quantization_config']
        self.quantization_stats = checkpoint['quantization_stats']
        self.quantization_parameters = checkpoint['quantization_parameters']
        print(f"📁 Quantized model loaded from {path}")
        return model


def create_quantization_config(
    target_compression: float = 4.0,
    accuracy_threshold: float = 0.05,
    use_mixed_precision: bool = False
) -> QuantizationConfig:
    """Create quantization configuration based on target compression and accuracy.
    
    Args:
        target_compression: Target compression ratio (e.g., 4.0 for 4x)
        accuracy_threshold: Maximum acceptable accuracy drop
        use_mixed_precision: Whether to use mixed precision quantization
        
    Returns:
        Optimized quantization configuration
    """
    if target_compression >= 8.0:
        # Aggressive quantization
        return QuantizationConfig(
            scheme=QuantizationScheme.INT4,
            strategy=LayerQuantizationStrategy.MIXED_PRECISION if use_mixed_precision else LayerQuantizationStrategy.UNIFORM,
            calibration_samples=1024,
            preserve_embeddings=False,
            mixed_precision_threshold=accuracy_threshold
        )
    elif target_compression >= 4.0:
        # Balanced quantization
        return QuantizationConfig(
            scheme=QuantizationScheme.INT8,
            strategy=LayerQuantizationStrategy.LAYER_WISE,
            calibration_samples=512,
            preserve_embeddings=True,
            mixed_precision_threshold=accuracy_threshold
        )
    else:
        # Conservative quantization
        return QuantizationConfig(
            scheme=QuantizationScheme.INT8,
            strategy=LayerQuantizationStrategy.SENSITIVITY_BASED,
            calibration_samples=256,
            preserve_embeddings=True,
            preserve_layer_norm=True,
            mixed_precision_threshold=accuracy_threshold
        )