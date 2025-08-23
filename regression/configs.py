"""
Copyright (c) 2025. All rights reserved.
"""

"""
Configuration classes for regression models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.configs import ModelConfig
from dataclasses import dataclass 

@dataclass 
class AutoregressiveDecodeConfig: 
    num_steps: int 
    expanding_context: bool
    max_seq_len: int

@dataclass 
class TransformerModelConfig(ModelConfig): 
    """Configuration for transformer model architecture."""
    max_seq_len: int 
    input_dim: int 
    embed_dim: int 
    ffn_latent_dim: int 
    num_layers: int 
    output_dim: int 
    num_heads: int 
    apply_causal_mask: bool
    autoregressive_mode: bool
    decode_config: AutoregressiveDecodeConfig

@dataclass 
class RegressionModelConfig(ModelConfig):
    """Configuration for neural network regression model architecture."""
    custom_act: str            # Activation function type
    num_latent_layers: int     # Number of hidden layers
    latent_dims: list[int]     # Hidden layer dimensions
    allow_residual: bool       # Enable residual connections


