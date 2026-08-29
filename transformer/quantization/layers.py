"""
Copyright (c) 2025. All rights reserved.
"""

"""
Quantized layer implementations for transformer models.

This module provides quantized versions of common transformer layers including
linear layers, embeddings, attention, and feed-forward networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


def quantize_tensor(tensor: torch.Tensor, scale: float, zero_point: int, 
                   dtype: torch.dtype = torch.qint8) -> torch.Tensor:
    """Quantize a tensor using given scale and zero point."""
    if dtype == torch.qint8:
        qmin, qmax = -128, 127
    elif dtype == torch.quint8:
        qmin, qmax = 0, 255
    elif dtype in [torch.qint32, torch.int8]:
        qmin, qmax = -2147483648, 2147483647
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Clamp and round
    quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
    return quantized.to(dtype)


def dequantize_tensor(qtensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Dequantize a tensor using given scale and zero point."""
    return scale * (qtensor.float() - zero_point)


class QuantizedLinear(nn.Module):
    """Quantized linear layer with INT8/INT4 support."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 8,
        activation_bits: int = 8,
        weight_scale: Optional[float] = None,
        weight_zero_point: Optional[int] = None,
        activation_scale: Optional[float] = None,
        activation_zero_point: Optional[int] = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        # Quantization parameters
        self.register_buffer('weight_scale', torch.tensor(weight_scale or 1.0))
        self.register_buffer('weight_zero_point', torch.tensor(weight_zero_point or 0, dtype=torch.int32))
        self.register_buffer('activation_scale', torch.tensor(activation_scale or 1.0))
        self.register_buffer('activation_zero_point', torch.tensor(activation_zero_point or 0, dtype=torch.int32))
        
        # Quantized weights
        if weight_bits == 8:
            self.register_buffer('quantized_weight', torch.zeros(out_features, in_features, dtype=torch.qint8))
        elif weight_bits == 4:
            # Pack two 4-bit weights per byte
            packed_size = (in_features + 1) // 2
            self.register_buffer('quantized_weight', torch.zeros(out_features, packed_size, dtype=torch.uint8))
        else:
            raise ValueError(f"Unsupported weight bits: {weight_bits}")
        
        if bias:
            self.register_buffer('quantized_bias', torch.zeros(out_features, dtype=torch.qint32))
            self.register_buffer('bias_scale', torch.tensor(1.0))
            self.register_buffer('bias_zero_point', torch.tensor(0, dtype=torch.int32))
        else:
            self.register_buffer('quantized_bias', None)
            
        self._is_quantized = False
        
    @classmethod
    def from_float(
        cls, 
        float_module: nn.Linear, 
        weight_stats: Dict[str, Any],
        activation_stats: Optional[Dict[str, Any]] = None,
        weight_bits: int = 8,
        activation_bits: int = 8
    ) -> 'QuantizedLinear':
        """Create quantized layer from float layer with calibration statistics."""
        
        # Calculate quantization parameters for weights
        weight_min = weight_stats.get('min', float_module.weight.min().item())
        weight_max = weight_stats.get('max', float_module.weight.max().item())
        
        if weight_bits == 8:
            weight_scale = (weight_max - weight_min) / 255.0
            weight_zero_point = int(round(-weight_min / weight_scale))
            weight_dtype = torch.qint8
        elif weight_bits == 4:
            weight_scale = (weight_max - weight_min) / 15.0
            weight_zero_point = int(round(-weight_min / weight_scale))
            weight_dtype = torch.quint8
        else:
            raise ValueError(f"Unsupported weight bits: {weight_bits}")
            
        # Calculate quantization parameters for activations
        if activation_stats:
            act_min = activation_stats.get('min', -6.0)
            act_max = activation_stats.get('max', 6.0)
        else:
            # Default activation range
            act_min, act_max = -6.0, 6.0
            
        if activation_bits == 8:
            activation_scale = (act_max - act_min) / 255.0
            activation_zero_point = int(round(-act_min / activation_scale))
        else:
            activation_scale = (act_max - act_min) / ((2 ** activation_bits) - 1)
            activation_zero_point = int(round(-act_min / activation_scale))
            
        # Create quantized layer
        quantized_layer = cls(
            in_features=float_module.in_features,
            out_features=float_module.out_features,
            bias=float_module.bias is not None,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            weight_scale=weight_scale,
            weight_zero_point=weight_zero_point,
            activation_scale=activation_scale,
            activation_zero_point=activation_zero_point
        )
        
        # Quantize weights
        quantized_layer._quantize_weights(float_module.weight, weight_dtype)
        
        # Quantize bias if present
        if float_module.bias is not None:
            bias_scale = weight_scale * activation_scale
            quantized_layer.bias_scale = torch.tensor(bias_scale)
            quantized_bias = torch.round(float_module.bias / bias_scale).to(torch.qint32)
            quantized_layer.quantized_bias = quantized_bias
            
        quantized_layer._is_quantized = True
        return quantized_layer
        
    def _quantize_weights(self, weight: torch.Tensor, dtype: torch.dtype):
        """Quantize and store weights."""
        if self.weight_bits == 8:
            quantized = quantize_tensor(weight, self.weight_scale.item(), 
                                      self.weight_zero_point.item(), dtype)
            self.quantized_weight = quantized
        elif self.weight_bits == 4:
            # Custom 4-bit quantization with packing
            scale = self.weight_scale.item()
            zero_point = self.weight_zero_point.item()
            
            # Quantize to 4-bit range [0, 15]
            quantized = torch.clamp(torch.round(weight / scale + zero_point), 0, 15)
            
            # Pack two 4-bit values per byte
            packed = torch.zeros(weight.shape[0], (weight.shape[1] + 1) // 2, dtype=torch.uint8)
            for i in range(weight.shape[1]):
                byte_idx = i // 2
                bit_offset = (i % 2) * 4
                packed[:, byte_idx] |= (quantized[:, i].to(torch.uint8) << bit_offset)
                
            self.quantized_weight = packed
            
    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights for computation."""
        if self.weight_bits == 8:
            return dequantize_tensor(self.quantized_weight, self.weight_scale.item(), 
                                   self.weight_zero_point.item())
        elif self.weight_bits == 4:
            # Unpack 4-bit weights
            scale = self.weight_scale.item()
            zero_point = self.weight_zero_point.item()
            
            unpacked = torch.zeros(self.out_features, self.in_features, dtype=torch.float32,
                                 device=self.quantized_weight.device)
            
            for i in range(self.in_features):
                byte_idx = i // 2
                bit_offset = (i % 2) * 4
                quantized_val = (self.quantized_weight[:, byte_idx] >> bit_offset) & 0xF
                unpacked[:, i] = scale * (quantized_val.float() - zero_point)
                
            return unpacked
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer not quantized. Call from_float() first.")
            
        # Dequantize weights for computation
        weight = self._dequantize_weights()
        
        # Dequantize bias if present
        bias = None
        if self.quantized_bias is not None:
            bias = dequantize_tensor(self.quantized_bias, self.bias_scale.item(), 
                                   self.bias_zero_point.item())
        
        return F.linear(input, weight, bias)


class QuantizedEmbedding(nn.Module):
    """Quantized embedding layer."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        weight_bits: int = 8,
        weight_scale: Optional[float] = None,
        weight_zero_point: Optional[int] = None
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight_bits = weight_bits
        
        # Quantization parameters
        self.register_buffer('weight_scale', torch.tensor(weight_scale or 1.0))
        self.register_buffer('weight_zero_point', torch.tensor(weight_zero_point or 0, dtype=torch.int32))
        
        # Quantized weights
        if weight_bits == 8:
            self.register_buffer('quantized_weight', 
                               torch.zeros(num_embeddings, embedding_dim, dtype=torch.qint8))
        elif weight_bits == 4:
            packed_size = (embedding_dim + 1) // 2
            self.register_buffer('quantized_weight', 
                               torch.zeros(num_embeddings, packed_size, dtype=torch.uint8))
        else:
            raise ValueError(f"Unsupported weight bits: {weight_bits}")
            
        self._is_quantized = False
        
    @classmethod
    def from_float(
        cls,
        float_module: nn.Embedding,
        weight_stats: Dict[str, Any],
        weight_bits: int = 8
    ) -> 'QuantizedEmbedding':
        """Create quantized embedding from float embedding."""
        
        # Calculate quantization parameters
        weight_min = weight_stats.get('min', float_module.weight.min().item())
        weight_max = weight_stats.get('max', float_module.weight.max().item())
        
        if weight_bits == 8:
            weight_scale = (weight_max - weight_min) / 255.0
            weight_zero_point = int(round(-weight_min / weight_scale))
            weight_dtype = torch.qint8
        elif weight_bits == 4:
            weight_scale = (weight_max - weight_min) / 15.0
            weight_zero_point = int(round(-weight_min / weight_scale))
            weight_dtype = torch.quint8
        else:
            raise ValueError(f"Unsupported weight bits: {weight_bits}")
            
        # Create quantized layer
        quantized_layer = cls(
            num_embeddings=float_module.num_embeddings,
            embedding_dim=float_module.embedding_dim,
            padding_idx=float_module.padding_idx,
            weight_bits=weight_bits,
            weight_scale=weight_scale,
            weight_zero_point=weight_zero_point
        )
        
        # Quantize weights
        quantized_layer._quantize_weights(float_module.weight, weight_dtype)
        quantized_layer._is_quantized = True
        
        return quantized_layer
        
    def _quantize_weights(self, weight: torch.Tensor, dtype: torch.dtype):
        """Quantize and store embedding weights."""
        if self.weight_bits == 8:
            quantized = quantize_tensor(weight, self.weight_scale.item(), 
                                      self.weight_zero_point.item(), dtype)
            self.quantized_weight = quantized
        elif self.weight_bits == 4:
            # 4-bit quantization with packing
            scale = self.weight_scale.item()
            zero_point = self.weight_zero_point.item()
            
            quantized = torch.clamp(torch.round(weight / scale + zero_point), 0, 15)
            
            # Pack weights
            packed = torch.zeros(weight.shape[0], (weight.shape[1] + 1) // 2, dtype=torch.uint8)
            for i in range(weight.shape[1]):
                byte_idx = i // 2
                bit_offset = (i % 2) * 4
                packed[:, byte_idx] |= (quantized[:, i].to(torch.uint8) << bit_offset)
                
            self.quantized_weight = packed
            
    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize embedding weights."""
        if self.weight_bits == 8:
            return dequantize_tensor(self.quantized_weight, self.weight_scale.item(),
                                   self.weight_zero_point.item())
        elif self.weight_bits == 4:
            # Unpack 4-bit weights
            scale = self.weight_scale.item()
            zero_point = self.weight_zero_point.item()
            
            unpacked = torch.zeros(self.num_embeddings, self.embedding_dim, dtype=torch.float32,
                                 device=self.quantized_weight.device)
            
            for i in range(self.embedding_dim):
                byte_idx = i // 2
                bit_offset = (i % 2) * 4
                quantized_val = (self.quantized_weight[:, byte_idx] >> bit_offset) & 0xF
                unpacked[:, i] = scale * (quantized_val.float() - zero_point)
                
            return unpacked
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer not quantized. Call from_float() first.")
            
        weight = self._dequantize_weights()
        return F.embedding(input, weight, self.padding_idx)


class QuantizedAttention(nn.Module):
    """Quantized multi-head attention layer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        weight_bits: int = 8,
        activation_bits: int = 8
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Quantized projection layers
        self.q_proj = QuantizedLinear(embed_dim, embed_dim, bias=bias, 
                                    weight_bits=weight_bits, activation_bits=activation_bits)
        self.k_proj = QuantizedLinear(embed_dim, embed_dim, bias=bias,
                                    weight_bits=weight_bits, activation_bits=activation_bits)
        self.v_proj = QuantizedLinear(embed_dim, embed_dim, bias=bias,
                                    weight_bits=weight_bits, activation_bits=activation_bits)
        self.out_proj = QuantizedLinear(embed_dim, embed_dim, bias=bias,
                                      weight_bits=weight_bits, activation_bits=activation_bits)
        
        self.dropout_layer = nn.Dropout(dropout)
        self._is_quantized = False
        
    @classmethod
    def from_float(
        cls,
        float_module: nn.MultiheadAttention,
        weight_stats: Dict[str, Dict[str, Any]],
        activation_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        weight_bits: int = 8,
        activation_bits: int = 8
    ) -> 'QuantizedAttention':
        """Create quantized attention from float attention."""
        
        quantized_attn = cls(
            embed_dim=float_module.embed_dim,
            num_heads=float_module.num_heads,
            dropout=float_module.dropout,
            bias=float_module.bias is not None,
            weight_bits=weight_bits,
            activation_bits=activation_bits
        )
        
        # Get activation stats if available
        act_stats = activation_stats or {}
        
        # Create quantized projection layers from float weights
        if hasattr(float_module, 'q_proj_weight') and float_module.q_proj_weight is not None:
            # Split combined weight matrix
            in_proj_weight = float_module.in_proj_weight
            embed_dim = float_module.embed_dim
            
            q_weight = in_proj_weight[:embed_dim]
            k_weight = in_proj_weight[embed_dim:2*embed_dim]
            v_weight = in_proj_weight[2*embed_dim:]
            
            # Create temporary linear modules for conversion
            q_linear = nn.Linear(embed_dim, embed_dim, bias=float_module.in_proj_bias is not None)
            k_linear = nn.Linear(embed_dim, embed_dim, bias=float_module.in_proj_bias is not None)
            v_linear = nn.Linear(embed_dim, embed_dim, bias=float_module.in_proj_bias is not None)
            out_linear = nn.Linear(embed_dim, embed_dim, bias=float_module.out_proj.bias is not None)
            
            q_linear.weight.data = q_weight
            k_linear.weight.data = k_weight
            v_linear.weight.data = v_weight
            out_linear.weight.data = float_module.out_proj.weight.data
            
            if float_module.in_proj_bias is not None:
                q_linear.bias.data = float_module.in_proj_bias[:embed_dim]
                k_linear.bias.data = float_module.in_proj_bias[embed_dim:2*embed_dim]
                v_linear.bias.data = float_module.in_proj_bias[2*embed_dim:]
                
            if float_module.out_proj.bias is not None:
                out_linear.bias.data = float_module.out_proj.bias.data
                
            # Convert to quantized layers
            quantized_attn.q_proj = QuantizedLinear.from_float(
                q_linear, weight_stats.get('q_proj', {}), 
                act_stats.get('q_proj'), weight_bits, activation_bits
            )
            quantized_attn.k_proj = QuantizedLinear.from_float(
                k_linear, weight_stats.get('k_proj', {}),
                act_stats.get('k_proj'), weight_bits, activation_bits
            )
            quantized_attn.v_proj = QuantizedLinear.from_float(
                v_linear, weight_stats.get('v_proj', {}),
                act_stats.get('v_proj'), weight_bits, activation_bits
            )
            quantized_attn.out_proj = QuantizedLinear.from_float(
                out_linear, weight_stats.get('out_proj', {}),
                act_stats.get('out_proj'), weight_bits, activation_bits
            )
            
        quantized_attn._is_quantized = True
        return quantized_attn
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._is_quantized:
            raise RuntimeError("Layer not quantized. Call from_float() first.")
            
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, embed_dim = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key) 
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores += attn_mask
            
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights.mean(dim=1)


class QuantizedFFN(nn.Module):
    """Quantized feed-forward network."""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        weight_bits: int = 8,
        activation_bits: int = 8
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        
        # Quantized linear layers
        self.linear1 = QuantizedLinear(embed_dim, ffn_dim, bias=True,
                                     weight_bits=weight_bits, activation_bits=activation_bits)
        self.linear2 = QuantizedLinear(ffn_dim, embed_dim, bias=True,
                                     weight_bits=weight_bits, activation_bits=activation_bits)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.dropout_layer = nn.Dropout(dropout)
        self._is_quantized = False
        
    @classmethod
    def from_float(
        cls,
        linear1: nn.Linear,
        linear2: nn.Linear,
        weight_stats: Dict[str, Dict[str, Any]],
        activation_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        dropout: float = 0.0,
        activation: str = "relu",
        weight_bits: int = 8,
        activation_bits: int = 8
    ) -> 'QuantizedFFN':
        """Create quantized FFN from float linear layers."""
        
        quantized_ffn = cls(
            embed_dim=linear1.in_features,
            ffn_dim=linear1.out_features,
            dropout=dropout,
            activation=activation,
            weight_bits=weight_bits,
            activation_bits=activation_bits
        )
        
        # Get activation stats if available
        act_stats = activation_stats or {}
        
        # Convert linear layers
        quantized_ffn.linear1 = QuantizedLinear.from_float(
            linear1, weight_stats.get('linear1', {}),
            act_stats.get('linear1'), weight_bits, activation_bits
        )
        quantized_ffn.linear2 = QuantizedLinear.from_float(
            linear2, weight_stats.get('linear2', {}),
            act_stats.get('linear2'), weight_bits, activation_bits
        )
        
        quantized_ffn._is_quantized = True
        return quantized_ffn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer not quantized. Call from_float() first.")
            
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    # Demo usage
    print("🔢 Quantized Layers Demo")
    print("=" * 50)
    
    # Create a simple float linear layer
    float_linear = nn.Linear(512, 256, bias=True)
    
    # Mock weight statistics
    weight_stats = {
        'min': float_linear.weight.min().item(),
        'max': float_linear.weight.max().item(),
        'mean': float_linear.weight.mean().item(),
        'std': float_linear.weight.std().item()
    }
    
    # Create quantized version
    quantized_linear = QuantizedLinear.from_float(
        float_linear, weight_stats, weight_bits=8, activation_bits=8
    )
    
    # Test forward pass
    input_tensor = torch.randn(32, 512)
    
    # Float output
    float_output = float_linear(input_tensor)
    
    # Quantized output
    quantized_output = quantized_linear(input_tensor)
    
    # Compare outputs
    mse_error = F.mse_loss(float_output, quantized_output)
    print(f"MSE between float and INT8: {mse_error:.6f}")
    
    # Memory comparison
    float_params = sum(p.numel() * p.element_size() for p in float_linear.parameters())
    quantized_params = sum(buf.numel() * buf.element_size() for buf in quantized_linear.buffers())
    
    print(f"Float model size: {float_params} bytes")
    print(f"Quantized model size: {quantized_params} bytes")
    print(f"Compression ratio: {float_params / quantized_params:.2f}x")
    
    print("\n✅ Quantized layers ready!")