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
from transformer.transformer_model import TransformerModel
from dataclasses import dataclass 
from configs import TransformerModelConfig

from typing import Optional, Tuple 
from lib.utils import set_seed
class TransformerRegressionModel(torch.nn.Module):
    """Regression wrapper for transformer model with synthetic data generation.""" 
    def __init__(self, config: TransformerModelConfig): 
        super().__init__()
        self.config = config
        self.model = TransformerModel(
            input_dim=config.input_dim, 
            embed_dim=config.embed_dim, 
            ffn_latent_dim=config.ffn_latent_dim, 
            num_layers=config.num_layers, 
            num_heads=config.num_heads, 
            output_dim=config.output_dim,
            causal_mask = config.apply_causal_mask,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Forward pass through transformer model."""
        return self.model(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]: 
        """Generate synthetic sequence data for transformer regression."""
        if random_seed: 
            set_seed(random_seed)

        # Generate 100 samples of input_dim=8 
        num_samples = 100 
        sequence_length = 32
        input_dim = self.config.input_dim
        x = torch.rand([num_samples, sequence_length, input_dim]) 
        y = torch.sum(x.reshape([num_samples, sequence_length * input_dim]), dim=1)
        return x, y    

if __name__ == "__main__": 
    model = TransformerModel(
        input_dim=8,
        embed_dim=32, 
        ffn_latent_dim=128, 
        num_layers=2, 
        num_heads=2, 
        output_dim=1
    )
    model.eval()
    output = model(input)