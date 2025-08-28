"""
Copyright (c) 2025. All rights reserved.
"""

"""
FP8 quantization module for transformer weights.

This module implements FP8 (8-bit floating point) quantization which provides
better dynamic range than INT8 while maintaining the same memory efficiency.
Supports E4M3 and E5M2 FP8 formats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum


class FP8Format(Enum):
    """FP8 format specifications."""
    E4M3 = "e4m3"  # 4 exponent bits, 3 mantissa bits
    E5M2 = "e5m2"  # 5 exponent bits, 2 mantissa bits


@dataclass
class FP8Config:
    """Configuration for FP8 quantization."""
    format: FP8Format = FP8Format.E4M3
    scale_method: str = "max"  # "max", "percentile", "mse"
    percentile: float = 99.9
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"  # "max", "most_recent"
    margin: float = 0  # Safety margin for scaling
    
    
class FP8Handler:
    """Handles FP8 number format conversions and operations."""
    
    def __init__(self, format: FP8Format = FP8Format.E4M3):
        self.format = format
        
        if format == FP8Format.E4M3:
            # E4M3: 1 sign + 4 exp + 3 mantissa
            # Exponents are 0 --> 15, 0 and 15 are reserved, max / min normal are computed with exp 14 and 1 
            self.exp_bits = 4
            self.mantissa_bits = 3
            self.exp_bias = 7
            # Max normal: exp=14 (1110), mantissa=111 (1.875)
            # Value = 1.875 × 2^(14-7) = 1.875 × 2^7 = 1.875 × 128 = 240
            self.max_normal = 240.0
            # Min normal: exp=1 (0001), mantissa=000 (1.0) 
            # Value = 1.0 × 2^(1-7) = 1.0 × 2^(-6) = 1/64
            self.min_normal = 1.0 / 64.0
        elif format == FP8Format.E5M2:
            # E5M2: 1 sign + 5 exp + 2 mantissa  
             # Exponents are 0 --> 31, 0 and 31 are reserved, max / min normal are computed with exp 30 and 1 
            self.exp_bits = 5
            self.mantissa_bits = 2
            self.exp_bias = 15
            # Max normal: exp=30 (11110), mantissa=11 (1.75)
            # Value = 1.75 × 2^(30-15) = 1.75 × 2^15 = 1.75 × 32768 = 57344
            self.max_normal = 57344.0
            # Min normal: exp=1 (00001), mantissa=00 (1.0)
            # Value = 1.0 × 2^(1-15) = 1.0 × 2^(-14) = 1/16384
            self.min_normal = 1.0 / 16384.0
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")
            
    def get_max_representable(self) -> float:
        """Get maximum representable value in this FP8 format."""
        return self.max_normal
        
    def quantize_tensor(self, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        """Quantize tensor to FP8 format and back to simulate FP8 storage."""
        # Scale tensor to FP8 range
        scaled_tensor = tensor / scale
        
        # Clamp to representable range
        max_val = self.max_normal
        clamped = torch.clamp(scaled_tensor, -max_val, max_val)
        
        # Simulate FP8 quantization by converting through numpy
        # (PyTorch doesn't have native FP8 support yet)
        if self.format == FP8Format.E4M3:
            # Simulate E4M3 precision
            quantized = self._simulate_e4m3_precision(clamped)
        else:
            # Simulate E5M2 precision
            quantized = self._simulate_e5m2_precision(clamped)
            
        # Scale back
        return quantized * scale
        
    def _simulate_e4m3_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate E4M3 precision by reducing mantissa bits.
        
        fp8_range = (input/scale).clamp(-max_normal, max_normal)
        fp8_range = fp8_range.float16.float32
        np_range = fp8_range.detach().cpu().numpy()
        quantized = np.round(np_range * 2**mantissa_bits)/2**mantissa_bits
        quantized = quantized.clip(-max_normal, max_normal)
        dequantized = quantized * scale 


        """
        # Convert to float32 numpy for manipulation
        np_tensor = tensor.detach().cpu().numpy()
        
        # Use float16 as approximation (closer to FP8 precision)
        # Then further reduce precision by rounding mantissa
        fp16_tensor = np_tensor.astype(np.float16).astype(np.float32)
        
        # Further reduce precision to simulate 3-bit mantissa
        # This is an approximation - actual FP8 would need hardware support
        scale_factor = 2 ** self.mantissa_bits
        quantized = np.round(fp16_tensor * scale_factor) / scale_factor
        
        # Clamp to E4M3 range
        quantized = np.clip(quantized, -self.max_normal, self.max_normal)
        
        return torch.tensor(quantized, dtype=tensor.dtype, device=tensor.device)
        
    def _simulate_e5m2_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate E5M2 precision by reducing mantissa bits.
    
        """
        np_tensor = tensor.detach().cpu().numpy()
        
        # Use float16 as base, then reduce mantissa precision to 2 bits
        fp16_tensor = np_tensor.astype(np.float16).astype(np.float32)
        
        # Reduce mantissa precision (2 bits = 4 levels)
        scale_factor = 4.0  # 2^2
        quantized = np.round(fp16_tensor * scale_factor) / scale_factor
        
        # Clamp to E5M2 range  
        quantized = np.clip(quantized, -self.max_normal, self.max_normal)
        
        return torch.tensor(quantized, dtype=tensor.dtype, device=tensor.device)


class FP8Quantizer:
    """Main FP8 quantizer for transformer weights."""
    
    def __init__(self, config: FP8Config = None):
        self.config = config or FP8Config()
        self.fp8_handler = FP8Handler(self.config.format)
        self.amax_history = []
        self.scale_cache = {}
        
    def compute_scale(self, tensor: torch.Tensor, name: str = "") -> float:
        """Compute scaling factor for FP8 quantization."""
        if self.config.scale_method == "max":
            amax = torch.max(torch.abs(tensor)).item()
        elif self.config.scale_method == "percentile":
            flat_tensor = tensor.flatten().abs()
            amax = torch.quantile(flat_tensor, self.config.percentile / 100.0).item()
        elif self.config.scale_method == "mse":
            # Find scale that minimizes MSE
            amax = self._compute_mse_optimal_scale(tensor)
        else:
            raise ValueError(f"Unknown scale method: {self.config.scale_method}")
            
        # Apply margin for safety
        amax = amax * (1.0 + self.config.margin)
        
        # Store in history
        if len(self.amax_history) >= self.config.amax_history_len:
            self.amax_history.pop(0)
        self.amax_history.append(amax)
        
        # Compute final amax based on algorithm
        if self.config.amax_compute_algo == "max":
            final_amax = max(self.amax_history)
        elif self.config.amax_compute_algo == "most_recent":
            final_amax = amax
        else:
            final_amax = amax
            
        # Compute scale
        # quantizer computes scale - 
        # How ? It calibrates, amax - max quantization, torch.quantile(tensor, percentile/100.00), mse comparison 
        # Once it computes amax / self.fp8_handler.geT_max_representable() = scale 

        max_representable = self.fp8_handler.get_max_representable()
        scale = final_amax / max_representable
        
        # Cache scale
        if name:
            self.scale_cache[name] = scale
            
        return max(scale, 1e-10)  # Avoid division by zero
        
    def _compute_mse_optimal_scale(self, tensor: torch.Tensor) -> float:
        """Find scale that minimizes MSE between original and quantized tensor."""
        tensor_abs = torch.abs(tensor)
        max_val = torch.max(tensor_abs).item()
        
        # Try different scales and find the one with minimum MSE
        best_scale = max_val / self.fp8_handler.get_max_representable()
        best_mse = float('inf')
        
        # Search around the initial estimate
        base_scale = max_val / self.fp8_handler.get_max_representable()
        search_range = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
        
        for factor in search_range:
            scale = base_scale * factor
            quantized = self.fp8_handler.quantize_tensor(tensor, scale)
            mse = F.mse_loss(tensor, quantized).item()
            
            if mse < best_mse:
                best_mse = mse
                best_scale = scale
                
        return best_scale * self.fp8_handler.get_max_representable()
        
    def quantize_weight(self, weight: torch.Tensor, name: str = "") -> Tuple[torch.Tensor, float]:
        """Quantize a weight tensor to FP8."""
        # What happens during quantization ? 
        # pass in tensor --> compute scale --> quantize_tensor 
        # compute scale - uses amax/max_representable
        # quantize - value / scale - fp16 --> fp32 --> simulate FP8 arithmetic 
        ## fp8_range = (input/scale).clamp(-max_normal, max_normal)
        # fp8_range = fp8_range.float16.float32
        # np_range = fp8_range.detach().cpu().numpy()
        # quantized = np.round(np_range * 2**mantissa_bits)/2**mantissa_bits
        # quantized = quantized.clip(-max_normal, max_normal)
        # dequantized = quantized * scale 

        scale = self.compute_scale(weight, name)
        quantized_weight = self.fp8_handler.quantize_tensor(weight, scale)
        return quantized_weight, scale
        
    def quantize_activation(self, activation: torch.Tensor, scale: float) -> torch.Tensor:
        """Quantize an activation tensor using pre-computed scale."""
        return self.fp8_handler.quantize_tensor(activation, scale)


class FP8Linear(nn.Module):
    """Linear layer with FP8 quantized weights."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fp8_config: FP8Config = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_config = fp8_config or FP8Config()
        self.quantizer = FP8Quantizer(self.fp8_config)
        
        # Original weights (for training)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Quantized weights and scale (for inference)
        self.register_buffer('quantized_weight', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('input_scale', None)
        
        self.quantized = False
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def quantize_weights(self):
        """Quantize the weights to FP8 format."""
        with torch.no_grad():
            quantized_weight, weight_scale = self.quantizer.quantize_weight(
                self.weight, f"linear_{id(self)}"
            )
            
            self.quantized_weight = quantized_weight
            self.weight_scale = torch.tensor(weight_scale)
            self.quantized = True
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantized and self.quantized_weight is not None:
            # Use quantized weights
            weight = self.quantized_weight # fake quantized. 
        else:
            # Use full precision weights
            weight = self.weight
            
        return F.linear(input, weight, self.bias)
        
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, fp8_format={self.fp8_config.format.value}, quantized={self.quantized}'


class FP8TransformerQuantizer:
    """Quantizes transformer models to FP8 format."""
    
    def __init__(self, config: FP8Config = None):
        self.config = config or FP8Config()
        self.quantizer = FP8Quantizer(self.config)
        
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize all applicable layers in the model to FP8."""
        self._replace_layers(model)
        self._quantize_weights(model)
        return model
        
    def _replace_layers(self, model: nn.Module):
        """Replace Linear layers with FP8Linear layers."""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Create FP8 replacement
                fp8_linear = FP8Linear(
                    module.in_features,
                    module.out_features, 
                    module.bias is not None,
                    self.config
                )
                
                # Copy weights
                with torch.no_grad():
                    fp8_linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        fp8_linear.bias.copy_(module.bias)
                        
                # Replace module
                setattr(model, name, fp8_linear)
            else:
                # Recursively process child modules
                self._replace_layers(module)
                
    def _quantize_weights(self, model: nn.Module):
        """Quantize weights in all FP8Linear layers."""
        for module in model.modules():
            if isinstance(module, FP8Linear):
                module.quantize_weights()
                
    def get_quantization_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get information about quantized model."""
        total_params = 0
        quantized_params = 0
        fp8_layers = 0
        
        for module in model.modules():
            if isinstance(module, FP8Linear):
                fp8_layers += 1
                layer_params = module.weight.numel()
                if module.bias is not None:
                    layer_params += module.bias.numel()
                    
                total_params += layer_params
                if module.quantized:
                    quantized_params += layer_params
                    
        return {
            'total_parameters': total_params,
            'quantized_parameters': quantized_params,
            'fp8_layers': fp8_layers,
            'quantization_ratio': quantized_params / total_params if total_params > 0 else 0,
            'fp8_format': self.config.format.value,
            'scale_method': self.config.scale_method
        }


def compare_fp8_formats(tensor: torch.Tensor) -> Dict[str, Any]:
    """Compare E4M3 and E5M2 FP8 formats on a given tensor."""
    original_mse = F.mse_loss(tensor, torch.zeros_like(tensor)).item()
    
    results = {}
    
    for format_type in [FP8Format.E4M3, FP8Format.E5M2]:
        config = FP8Config(format=format_type)
        quantizer = FP8Quantizer(config)
        
        # Quantize tensor
        quantized, scale = quantizer.quantize_weight(tensor)
        
        # Compute metrics
        mse = F.mse_loss(tensor, quantized).item()
        mae = F.l1_loss(tensor, quantized).item()
        max_error = torch.max(torch.abs(tensor - quantized)).item()
        
        # Relative metrics
        rel_mse = mse / (original_mse + 1e-10)
        
        results[format_type.value] = {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'relative_mse': rel_mse,
            'scale': scale,
            'max_representable': quantizer.fp8_handler.get_max_representable()
        }
        
    return results


if __name__ == "__main__":
    # Demo usage
    print("🔢 FP8 Quantization Demo")
    print("=" * 50)
    
    # Create a simple transformer layer
    model = nn.Sequential(
        nn.Linear(512, 2048),
        nn.GELU(),
        nn.Linear(2048, 512)
    )
    
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Quantize to FP8
    fp8_quantizer = FP8TransformerQuantizer(
        FP8Config(format=FP8Format.E4M3, scale_method="percentile")
    )
    
    quantized_model = fp8_quantizer.quantize_model(model)
    
    # Get quantization info
    info = fp8_quantizer.get_quantization_info(quantized_model)
    
    print(f"\nFP8 Quantization Results:")
    print(f"  Format: {info['fp8_format']}")
    print(f"  FP8 layers: {info['fp8_layers']}")
    print(f"  Quantized parameters: {info['quantized_parameters']:,}")
    print(f"  Quantization ratio: {info['quantization_ratio']:.2%}")
    
    # Test inference
    input_tensor = torch.randn(32, 512)
    
    with torch.no_grad():
        original_output = model(input_tensor)
        quantized_output = quantized_model(input_tensor)
        
    # Compute error
    mse_error = F.mse_loss(original_output, quantized_output)
    cosine_sim = F.cosine_similarity(
        original_output.flatten(), quantized_output.flatten(), dim=0
    )
    
    print(f"\nInference Comparison:")
    print(f"  MSE Error: {mse_error:.6f}")
    print(f"  Cosine Similarity: {cosine_sim:.6f}")
    
    # Compare FP8 formats
    test_weights = torch.randn(256, 512) * 0.1
    format_comparison = compare_fp8_formats(test_weights)
    
    print(f"\nFormat Comparison:")
    for format_name, metrics in format_comparison.items():
        print(f"  {format_name.upper()}:")
        print(f"    MSE: {metrics['mse']:.8f}")
        print(f"    Max Error: {metrics['max_error']:.6f}")
        print(f"    Max Representable: {metrics['max_representable']:.1f}")
    
    print("\n✅ FP8 quantization ready!")