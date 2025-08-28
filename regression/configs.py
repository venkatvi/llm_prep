"""
Copyright (c) 2025. All rights reserved.
"""

"""
Configuration classes for regression models.
"""

import os
import sys
from dataclasses import dataclass
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.configs import ModelConfig
from transformer.configs import FFNConfig

@dataclass
class AutoregressiveDecodeConfig:
    """Configuration for autoregressive decoding in sequence models.

    Attributes:
        num_steps (int): Number of autoregressive decoding steps
        expanding_context (bool): Whether to use expanding context during decoding
        max_seq_len (int): Maximum sequence length for generation
    """

    num_steps: int
    expanding_context: bool
    max_seq_len: int
    use_kv_cache: bool = True


@dataclass
class TransformerModelConfig(ModelConfig):
    """Configuration for transformer model architecture.

    Attributes:
        max_seq_len (int): Maximum input sequence length
        input_dim (int): Input feature dimension
        embed_dim (int): Model embedding dimension
        ffn_latent_dim (int): Feed-forward network hidden dimension
        num_layers (int): Number of transformer layers
        output_dim (int): Output dimension
        num_heads (int): Number of attention heads
        num_groups (int): Number of key/value groups for GQA (Group Query Attention)
        apply_causal_mask (bool): Whether to apply causal masking for autoregressive generation
        autoregressive_mode (bool): Whether model operates in autoregressive mode
        decode_config (AutoregressiveDecodeConfig): Configuration for autoregressive decoding
    """

    max_seq_len: int
    input_dim: int
    embed_dim: int
    ffn_latent_dim: int
    num_layers: int
    output_dim: int
    num_heads: int
    num_groups: int
    apply_causal_mask: bool
    autoregressive_mode: bool
    decode_config: AutoregressiveDecodeConfig
    attention_type: str
    ffn_config: FFNConfig
    vocab_size: int = 0  # Optional, for future use


@dataclass
class EncoderDecoderConfig(ModelConfig):
    """Configuration for encoder-decoder transformer architecture.

    Attributes:
        max_seq_len (int): Maximum input/output sequence length
        input_dim (int): Input feature dimension
        embed_dim (int): Model embedding dimension
        ffn_latent_dim (int): Feed-forward network hidden dimension
        num_encoder_layers (int): Number of encoder transformer layers
        num_decoder_layers (int): Number of decoder transformer layers
        output_dim (int): Output dimension
        num_heads (int): Number of attention heads in each layer
        num_groups (int): Number of key/value groups for GQA (Group Query Attention)
        apply_causal_mask (bool): Whether to apply causal masking in decoder
        autoregressive_mode (bool): Whether decoder operates autoregressively
        decode_config (AutoregressiveDecodeConfig): Configuration for autoregressive decoding
    """

    max_seq_len: int
    input_dim: int
    embed_dim: int
    ffn_latent_dim: int
    num_encoder_layers: int
    num_decoder_layers: int
    output_dim: int
    num_heads: int
    num_groups: int
    apply_causal_mask: bool
    autoregressive_mode: bool
    decode_config: AutoregressiveDecodeConfig
    attention_type: str
    ffn_config: FFNConfig



@dataclass
class RegressionModelConfig(ModelConfig):
    """Configuration for neural network regression model architecture.

    Attributes:
        custom_act (str): Activation function type (e.g., 'relu', 'tanh', 'sigmoid')
        num_latent_layers (int): Number of hidden layers in the network
        latent_dims (List[int]): List of hidden layer dimensions, must match num_latent_layers
        allow_residual (bool): Whether to enable residual connections in activation layers
    """

    custom_act: str  # Activation function type
    num_latent_layers: int  # Number of hidden layers
    latent_dims: List[int]  # Hidden layer dimensions
    allow_residual: bool  # Enable residual connections
