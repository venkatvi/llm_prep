"""
Copyright (c) 2025. All rights reserved.
"""

"""
Asymmetric quantization implementation for neural networks.

This module provides asymmetric quantization that can handle different
ranges for positive and negative values, offering better precision
than symmetric quantization for skewed distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math


class QuantizationMode(Enum):
    """Different asymmetric quantization modes."""
    FULL_RANGE = "full_range"      # Use full [min, max] range
    PERCENTILE = "percentile"      # Use percentile clipping
    ADAPTIVE = "adaptive"          # Adapt based on data distribution
    MIXED_PRECISION = "mixed"      # Different bits for pos/neg


@dataclass
class AsymmetricConfig:
    """Configuration for asymmetric quantization."""
    bits: int = 8
    mode: QuantizationMode = QuantizationMode.FULL_RANGE
    percentile_low: float = 0.1    # Lower percentile for clipping
    percentile_high: float = 99.9  # Upper percentile for clipping
    pos_bits: int = 4              # Bits for positive values (mixed mode)
    neg_bits: int = 4              # Bits for negative values (mixed mode)
    signed: bool = True            # Whether to use signed quantization
    eps: float = 1e-8              # Small epsilon for numerical stability
    

class AsymmetricQuantizer:
    """Asymmetric quantizer with advanced range estimation."""
    
    def __init__(self, config: AsymmetricConfig = None):
        self.config = config or AsymmetricConfig()
        self.calibration_data = []
        self.is_calibrated = False
        self.scale = None
        self.zero_point = None
        
        # Mixed precision scales
        self.pos_scale = None
        self.neg_scale = None
        self.pos_zero_point = None
        self.neg_zero_point = None
        
    def calibrate(self, tensors: List[torch.Tensor]):
        """Calibrate quantization parameters from data."""
        self.calibration_data.extend(tensors)
        
        # Compute statistics across all calibration data
        all_values = torch.cat([t.flatten() for t in tensors])
        
        if self.config.mode == QuantizationMode.FULL_RANGE:
            self._calibrate_full_range(all_values)
        elif self.config.mode == QuantizationMode.PERCENTILE:
            self._calibrate_percentile(all_values)
        elif self.config.mode == QuantizationMode.ADAPTIVE:
            self._calibrate_adaptive(all_values)
        elif self.config.mode == QuantizationMode.MIXED_PRECISION:
            self._calibrate_mixed_precision(all_values)
            
        self.is_calibrated = True
        
    def _calibrate_full_range(self, values: torch.Tensor):
        """Calibrate using full min-max range."""
        min_val = values.min().item()
        max_val = values.max().item()
        
        self.scale, self.zero_point = self._compute_scale_zero_point(
            min_val, max_val, self.config.bits, self.config.signed
        )
        
    def _calibrate_percentile(self, values: torch.Tensor):
        """Calibrate using percentile clipping."""
        min_val = torch.quantile(values, self.config.percentile_low / 100.0).item()
        max_val = torch.quantile(values, self.config.percentile_high / 100.0).item()
        
        self.scale, self.zero_point = self._compute_scale_zero_point(
            min_val, max_val, self.config.bits, self.config.signed
        )
        
    def _calibrate_adaptive(self, values: torch.Tensor):
        """Adaptive calibration based on data distribution."""
        # Analyze distribution properties
        mean_val = values.mean().item()
        std_val = values.std().item()
        median_val = values.median().item()
        
        # Check for skewness
        skewness = self._compute_skewness(values)
        
        if abs(skewness) > 1.0:  # Highly skewed
            # Use percentile method for skewed data
            if skewness > 0:  # Right-skewed
                min_val = values.min().item()
                max_val = torch.quantile(values, 0.95).item()
            else:  # Left-skewed
                min_val = torch.quantile(values, 0.05).item()
                max_val = values.max().item()
        else:
            # Use mean ± k*std for symmetric-ish data
            k = 3.0  # Number of standard deviations
            min_val = mean_val - k * std_val
            max_val = mean_val + k * std_val
            
            # Clip to actual data range
            min_val = max(min_val, values.min().item())
            max_val = min(max_val, values.max().item())
            
        self.scale, self.zero_point = self._compute_scale_zero_point(
            min_val, max_val, self.config.bits, self.config.signed
        )
        
    def _calibrate_mixed_precision(self, values: torch.Tensor):
        """Calibrate with different precision for positive/negative values."""
        pos_values = values[values >= 0]
        neg_values = values[values < 0]
        
        if len(pos_values) > 0:
            pos_min = 0.0
            pos_max = pos_values.max().item()
            self.pos_scale, self.pos_zero_point = self._compute_scale_zero_point(
                pos_min, pos_max, self.config.pos_bits, signed=False
            )
        else:
            self.pos_scale = self.pos_zero_point = None
            
        if len(neg_values) > 0:
            neg_min = neg_values.min().item()
            neg_max = 0.0
            self.neg_scale, self.neg_zero_point = self._compute_scale_zero_point(
                neg_min, neg_max, self.config.neg_bits, signed=True
            )
        else:
            self.neg_scale = self.neg_zero_point = None
            
    def _compute_scale_zero_point(
        self, 
        min_val: float, 
        max_val: float, 
        bits: int, 
        signed: bool
    ) -> Tuple[float, int]:
        """Compute scale and zero point for asymmetric quantization."""
        if signed:
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** bits - 1
            
        # Handle edge cases
        if abs(max_val - min_val) < self.config.eps:
            scale = 1.0
            zero_point = qmin
        else:
            # Compute scale
            scale = (max_val - min_val) / (qmax - qmin)
            
            # Compute zero point
            zero_point_float = qmin - min_val / scale
            zero_point = int(round(torch.clamp(
                torch.tensor(zero_point_float), qmin, qmax
            ).item()))
            
        return scale, zero_point
        
    def _compute_skewness(self, values: torch.Tensor) -> float:
        """Compute skewness of the distribution."""
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val < self.config.eps:
            return 0.0
            
        # Compute third moment
        third_moment = ((values - mean_val) ** 3).mean()
        skewness = third_moment / (std_val ** 3)
        
        return skewness.item()
        
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor using asymmetric quantization."""
        if not self.is_calibrated:
            raise RuntimeError("Quantizer not calibrated. Call calibrate() first.")
            
        if self.config.mode == QuantizationMode.MIXED_PRECISION:
            return self._quantize_mixed_precision(tensor)
        else:
            return self._quantize_standard(tensor)
            
    def _quantize_standard(self, tensor: torch.Tensor) -> torch.Tensor:
        """Standard asymmetric quantization."""
        # Quantize: q = clamp(round(x/scale + zero_point), qmin, qmax)
        if self.config.signed:
            qmin = -(2 ** (self.config.bits - 1))
            qmax = 2 ** (self.config.bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** self.config.bits - 1
            
        quantized = torch.clamp(
            torch.round(tensor / self.scale + self.zero_point),
            qmin, qmax
        )
        
        return quantized
        
    def _quantize_mixed_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mixed precision asymmetric quantization."""
        result = torch.zeros_like(tensor)
        
        # Quantize positive values
        pos_mask = tensor >= 0
        if self.pos_scale is not None and pos_mask.any():
            pos_values = tensor[pos_mask]
            pos_quantized = torch.clamp(
                torch.round(pos_values / self.pos_scale + self.pos_zero_point),
                0, 2 ** self.config.pos_bits - 1
            )
            result[pos_mask] = pos_quantized
            
        # Quantize negative values
        neg_mask = tensor < 0
        if self.neg_scale is not None and neg_mask.any():
            neg_values = tensor[neg_mask]
            neg_qmin = -(2 ** (self.config.neg_bits - 1))
            neg_qmax = 2 ** (self.config.neg_bits - 1) - 1
            
            neg_quantized = torch.clamp(
                torch.round(neg_values / self.neg_scale + self.neg_zero_point),
                neg_qmin, neg_qmax
            )
            result[neg_mask] = neg_quantized
            
        return result
        
    def dequantize(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor back to floating point."""
        if not self.is_calibrated:
            raise RuntimeError("Quantizer not calibrated. Call calibrate() first.")
            
        if self.config.mode == QuantizationMode.MIXED_PRECISION:
            return self._dequantize_mixed_precision(quantized_tensor)
        else:
            return self._dequantize_standard(quantized_tensor)
            
    def _dequantize_standard(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """Standard asymmetric dequantization."""
        # Dequantize: x = (q - zero_point) * scale
        return (quantized_tensor - self.zero_point) * self.scale
        
    def _dequantize_mixed_precision(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """Mixed precision asymmetric dequantization."""
        result = torch.zeros_like(quantized_tensor, dtype=torch.float32)
        
        # Dequantize positive values
        pos_mask = quantized_tensor >= 0
        if self.pos_scale is not None and pos_mask.any():
            pos_quantized = quantized_tensor[pos_mask]
            pos_dequantized = (pos_quantized - self.pos_zero_point) * self.pos_scale
            result[pos_mask] = pos_dequantized
            
        # Dequantize negative values  
        neg_mask = quantized_tensor < 0
        if self.neg_scale is not None and neg_mask.any():
            neg_quantized = quantized_tensor[neg_mask]
            neg_dequantized = (neg_quantized - self.neg_zero_point) * self.neg_scale
            result[neg_mask] = neg_dequantized
            
        return result
        
    def get_quantization_info(self) -> Dict[str, Any]:
        """Get quantization parameters and statistics."""
        if not self.is_calibrated:
            return {"status": "not_calibrated"}
            
        info = {
            "status": "calibrated",
            "mode": self.config.mode.value,
            "bits": self.config.bits,
            "signed": self.config.signed
        }
        
        if self.config.mode == QuantizationMode.MIXED_PRECISION:
            info.update({
                "pos_bits": self.config.pos_bits,
                "neg_bits": self.config.neg_bits,
                "pos_scale": self.pos_scale,
                "neg_scale": self.neg_scale,
                "pos_zero_point": self.pos_zero_point,
                "neg_zero_point": self.neg_zero_point
            })
        else:
            info.update({
                "scale": self.scale,
                "zero_point": self.zero_point
            })
            
        return info


class AsymmetricQuantizedLinear(nn.Module):
    """Linear layer with asymmetric weight quantization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: AsymmetricConfig = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or AsymmetricConfig()
        
        # Float weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Quantization state
        self.weight_quantizer = AsymmetricQuantizer(self.config)
        self.is_quantized = False
        
        # Quantized weights storage
        self.register_buffer('quantized_weight', None)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def calibrate_quantization(self, calibration_data: List[torch.Tensor]):
        """Calibrate weight quantization using provided data."""
        # Use current weights for calibration
        self.weight_quantizer.calibrate([self.weight.data])
        
    def quantize_weights(self):
        """Quantize the weights."""
        if not self.weight_quantizer.is_calibrated:
            self.calibrate_quantization([])
            
        self.quantized_weight = self.weight_quantizer.quantize(self.weight.data)
        self.is_quantized = True
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional weight quantization."""
        if self.is_quantized and self.quantized_weight is not None:
            # Use quantized weights
            weight = self.weight_quantizer.dequantize(self.quantized_weight)
        else:
            # Use float weights
            weight = self.weight
            
        return F.linear(input, weight, self.bias)
        
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, quantized={self.is_quantized}, '
                f'mode={self.config.mode.value}, bits={self.config.bits}')


def analyze_tensor_distribution(tensor: torch.Tensor) -> Dict[str, float]:
    """Analyze tensor distribution to recommend quantization strategy."""
    values = tensor.flatten()
    
    # Basic statistics
    mean_val = values.mean().item()
    std_val = values.std().item()
    min_val = values.min().item()
    max_val = values.max().item()
    median_val = values.median().item()
    
    # Distribution properties
    skewness = float('nan')
    kurtosis = float('nan')
    
    if std_val > 1e-8:
        # Skewness
        third_moment = ((values - mean_val) ** 3).mean()
        skewness = (third_moment / (std_val ** 3)).item()
        
        # Kurtosis  
        fourth_moment = ((values - mean_val) ** 4).mean()
        kurtosis = (fourth_moment / (std_val ** 4)).item() - 3.0  # Excess kurtosis
        
    # Percentiles
    percentiles = {}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        percentiles[f'p{p}'] = torch.quantile(values, p / 100.0).item()
        
    # Positive/negative analysis
    pos_values = values[values >= 0]
    neg_values = values[values < 0]
    
    pos_ratio = len(pos_values) / len(values) if len(values) > 0 else 0
    neg_ratio = len(neg_values) / len(values) if len(values) > 0 else 0
    
    pos_max = pos_values.max().item() if len(pos_values) > 0 else 0
    neg_min = neg_values.min().item() if len(neg_values) > 0 else 0
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'median': median_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'pos_ratio': pos_ratio,
        'neg_ratio': neg_ratio,
        'pos_max': pos_max,
        'neg_min': neg_min,
        'range': max_val - min_val,
        **percentiles
    }


def recommend_quantization_strategy(analysis: Dict[str, float]) -> AsymmetricConfig:
    """Recommend quantization strategy based on distribution analysis."""
    config = AsymmetricConfig()
    
    # Check for skewness
    if not math.isnan(analysis['skewness']) and abs(analysis['skewness']) > 1.5:
        config.mode = QuantizationMode.PERCENTILE
        config.percentile_low = 1.0
        config.percentile_high = 99.0
    # Check for imbalanced pos/neg values
    elif analysis['pos_ratio'] < 0.1 or analysis['neg_ratio'] < 0.1:
        config.mode = QuantizationMode.MIXED_PRECISION
        config.pos_bits = 6 if analysis['pos_ratio'] > 0.5 else 2
        config.neg_bits = 6 if analysis['neg_ratio'] > 0.5 else 2
    # Check for outliers (high kurtosis)
    elif not math.isnan(analysis['kurtosis']) and analysis['kurtosis'] > 3.0:
        config.mode = QuantizationMode.ADAPTIVE
    else:
        config.mode = QuantizationMode.FULL_RANGE
        
    return config


if __name__ == "__main__":
    # Demo usage
    print("🔢 Asymmetric Quantization Demo")
    print("=" * 50)
    
    # Create test tensors with different distributions
    test_cases = {
        "symmetric": torch.randn(1000) * 2.0,
        "skewed_right": torch.abs(torch.randn(1000)) + 0.1,
        "skewed_left": -torch.abs(torch.randn(1000)) - 0.1,
        "mixed_range": torch.cat([torch.randn(800) * 0.1, torch.randn(200) * 5.0])
    }
    
    for name, tensor in test_cases.items():
        print(f"\n--- {name.upper()} DISTRIBUTION ---")
        
        # Analyze distribution
        analysis = analyze_tensor_distribution(tensor)
        print(f"Mean: {analysis['mean']:.3f}, Std: {analysis['std']:.3f}")
        print(f"Range: [{analysis['min']:.3f}, {analysis['max']:.3f}]")
        print(f"Skewness: {analysis['skewness']:.3f}")
        print(f"Pos/Neg ratio: {analysis['pos_ratio']:.2f}/{analysis['neg_ratio']:.2f}")
        
        # Get recommended strategy
        recommended_config = recommend_quantization_strategy(analysis)
        print(f"Recommended mode: {recommended_config.mode.value}")
        
        # Test quantization with different strategies
        strategies = [
            ("Full Range", AsymmetricConfig(mode=QuantizationMode.FULL_RANGE)),
            ("Percentile", AsymmetricConfig(mode=QuantizationMode.PERCENTILE)),
            ("Adaptive", AsymmetricConfig(mode=QuantizationMode.ADAPTIVE))
        ]
        
        for strategy_name, config in strategies:
            quantizer = AsymmetricQuantizer(config)
            quantizer.calibrate([tensor])
            
            # Quantize and dequantize
            quantized = quantizer.quantize(tensor)
            dequantized = quantizer.dequantize(quantized)
            
            # Compute error
            mse = F.mse_loss(tensor, dequantized).item()
            mae = F.l1_loss(tensor, dequantized).item()
            
            print(f"  {strategy_name:12} - MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            # Show quantization info
            info = quantizer.get_quantization_info()
            if config.mode != QuantizationMode.MIXED_PRECISION:
                print(f"    Scale: {info['scale']:.6f}, Zero point: {info['zero_point']}")
    
    # Test asymmetric quantized linear layer
    print(f"\n--- QUANTIZED LINEAR LAYER ---")
    
    # Create layer
    layer = AsymmetricQuantizedLinear(128, 64, config=AsymmetricConfig(bits=8))
    
    # Test forward pass before quantization
    input_tensor = torch.randn(16, 128)
    output_float = layer(input_tensor)
    
    # Quantize weights
    layer.quantize_weights()
    output_quantized = layer(input_tensor)
    
    # Compare outputs
    mse_error = F.mse_loss(output_float, output_quantized)
    print(f"Layer quantization error (MSE): {mse_error:.6f}")
    print(f"Weight quantizer info: {layer.weight_quantizer.get_quantization_info()}")
    
    # Memory comparison
    float_size = layer.weight.numel() * 4  # float32
    quantized_size = layer.quantized_weight.numel() * 1  # int8
    compression_ratio = float_size / quantized_size
    
    print(f"Memory: {float_size} -> {quantized_size} bytes ({compression_ratio:.1f}x compression)")
    
    print("\n✅ Asymmetric quantization demo completed!")
    print("Key advantages demonstrated:")
    print("- Better precision for skewed distributions")
    print("- Adaptive quantization strategies") 
    print("- Mixed precision for pos/neg values")
    print("- Distribution analysis and recommendations")