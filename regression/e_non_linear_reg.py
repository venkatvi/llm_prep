"""
Multi-Layer Perceptron for non-linear regression.
"""
import os
import sys
from typing import Optional, Tuple

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.activations import get_activation_layer
from lib.configs import ModelConfig
from lib.utils import set_seed


class MLP(torch.nn.Module):
    """MLP with configurable layers, activations, and optional residual connections."""
    
    config: ModelConfig
    layers: torch.nn.ModuleList
    output_layer: torch.nn.Linear
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        
        assert len(self.config.latent_dims) == self.config.num_latent_layers, \
            f"Number of dimensions ({len(self.config.latent_dims)}) must match number of layers ({self.config.num_latent_layers})"
        
        output_dim: int = 1
        for layer_index in range(self.config.num_latent_layers):
            input_dim: int = 1 if layer_index == 0 else output_dim
            output_dim = self.config.latent_dims[layer_index]
            
            linear_layer: torch.nn.Linear = torch.nn.Linear(input_dim, output_dim)
            self.layers.append(linear_layer)

            act_layer: torch.nn.Module = get_activation_layer(self.config.custom_act)
            self.layers.append(act_layer)
        
        self.output_layer = torch.nn.Linear(output_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if not isinstance(layer, torch.nn.Linear) and self.config.allow_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        
        return self.output_layer(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic non-linear data: y = 4x² + 2x + noise"""
        if random_seed:
            set_seed(random_seed)
        
        # Generate random inputs in range [0, 10]
        inputs: torch.Tensor = torch.rand(100, 1) * 10
        
        # Generate targets with quadratic relationship y = 4x² + 2x + noise
        targets: torch.Tensor = 4 * inputs**2 + 2 * inputs + torch.rand(100, 1)
        
        return inputs, targets