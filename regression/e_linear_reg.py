"""
Linear regression model: y = Wx + b with optional activation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Optional, Tuple

from lib.activations import get_activation_layer
from lib.configs import ModelConfig
from lib.utils import set_seed
class LinearRegressionModel(torch.nn.Module):
    """Linear regression with optional activation function."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        
        if config.custom_act is not None:
            self.activation_layer = get_activation_layer(config.custom_act)
        else: 
            self.activation_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.activation_layer(x) if self.activation_layer else x 

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data: y = 100x + noise"""
        if random_seed: 
            set_seed(random_seed)
        
        # Generate random inputs in range [0, 10]
        inputs = torch.rand(100, 1) * 10
        
        # Generate targets with linear relationship y = 100x + noise
        targets = 100 * inputs + torch.rand(100, 1)
        
        return inputs, targets