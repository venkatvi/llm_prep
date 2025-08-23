"""
Copyright (c) 2025. All rights reserved.
"""

"""
Transformer-based regression model wrapper.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformer.transformer_model import TransformerModel, AutoregressiveTransformerModel
from dataclasses import dataclass
from configs import TransformerModelConfig

from typing import Optional, Tuple
from lib.utils import set_seed


class RegressionTransformerModel(torch.nn.Module):
    """Regression wrapper for transformer model with synthetic data generation.
    
    This model uses global average pooling to produce a single output value per sample,
    making it suitable for regression tasks where we need to predict a scalar from a sequence.
    """
    
    config: TransformerModelConfig
    model: TransformerModel
    
    def __init__(self, config: TransformerModelConfig) -> None: 
        super().__init__()
        self.config = config
        self.model = TransformerModel(
            input_dim=config.input_dim, 
            embed_dim=config.embed_dim, 
            ffn_latent_dim=config.ffn_latent_dim, 
            num_layers=config.num_layers, 
            num_heads=config.num_heads, 
            output_dim=config.output_dim,
            apply_causal_mask=config.apply_causal_mask,
            max_seq_len=config.max_seq_len
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Forward pass through transformer model."""
        return self.model(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]: 
        """Generate synthetic sequence data for transformer regression.
        
        Creates random input sequences where the target is the sum of all elements,
        providing a simple regression task that requires understanding the full sequence.
        
        Args:
            random_seed: Optional seed for reproducible data generation
            
        Returns:
            Tuple of (input_sequences, targets) where:
            - input_sequences: [num_samples, seq_len, input_dim] 
            - targets: [num_samples] scalar regression targets
        """
        if random_seed:
            set_seed(random_seed)

        # Generate 100 samples with specified sequence length and input dimension
        num_samples: int = 100
        sequence_length: int = 32
        input_dim: int = self.config.input_dim
        x: torch.Tensor = torch.rand([num_samples, sequence_length, input_dim])
        # Target is sum of all sequence elements (regression task)
        y: torch.Tensor = torch.sum(x.reshape([num_samples, sequence_length * input_dim]), dim=1)
        return x, y    

class ARTransformerModel(torch.nn.Module):
    """Autoregressive transformer wrapper for sequence-to-sequence prediction.
    
    This model generates data in an autoregressive format where each input sequence
    is paired with the next token in the sequence, enabling next-token prediction training.
    Unlike the regression model, this preserves sequence structure for generative tasks.
    """
    
    config: TransformerModelConfig
    model: AutoregressiveTransformerModel
    
    def __init__(self, config: TransformerModelConfig) -> None: 
        super().__init__()
        self.config = config
        self.model = AutoregressiveTransformerModel(
            input_dim=config.input_dim,
            embed_dim=config.embed_dim,
            ffn_latent_dim=config.ffn_latent_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads, 
            output_dim=config.output_dim,
            apply_causal_mask=config.apply_causal_mask,
            max_seq_len=config.max_seq_len
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Forward pass through transformer model."""
        return self.model(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]: 
        """Generate synthetic sequence data for autoregressive training.
        
        Creates input-target pairs where each target is the next token in the sequence,
        enabling the model to learn next-token prediction for autoregressive generation.
        
        Args:
            random_seed: Optional seed for reproducible data generation
            
        Returns:
            Tuple of (input_sequences, target_sequences) where:
            - input_sequences: [num_samples, seq_len-1, input_dim] current tokens
            - target_sequences: [num_samples, seq_len-1, input_dim] next tokens
        """
        if random_seed:
            set_seed(random_seed)

        # Generate 101 samples to create 100 input-target pairs after shifting
        num_samples: int = 101
        sequence_length: int = 32
        input_dim: int = self.config.input_dim
        sample_sequences: torch.Tensor = torch.rand([num_samples, sequence_length, input_dim])
        # Create autoregressive pairs: input[:-1] -> target[1:]
        x: torch.Tensor = sample_sequences[:, :-1, :]  # All tokens except last
        y: torch.Tensor = sample_sequences[:, 1:, :]   # All tokens except first
        return x, y

