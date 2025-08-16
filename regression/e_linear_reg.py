"""
Copyright (c) 2025. All rights reserved.
"""

"""
Linear regression model implementation.

This module implements a simple linear regression model using PyTorch's nn.Module.
The model learns a linear relationship y = ax + b from input data.
"""

import torch
from typing import Optional, Tuple

from activations import get_activation_layer
from configs import ModelConfig
class LinearRegressionModel(torch.nn.Module):
    """
    Linear regression model with optional activation function.
    
    This model implements a simple linear transformation y = Wx + b where W is a weight
    matrix and b is a bias term. An optional activation function can be applied after
    the linear transformation for non-linear behavior.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the linear regression model.
        
        Args:
            custom_act (str, optional): Name of activation function to apply after linear layer.
                                      If None, no activation is applied (pure linear model).
                                      Supported activations: relu, tanh, sigmoid, leakyrelu, gelu, silu
        """
        super().__init__()
        # Single input to single output linear layer
        self.linear = torch.nn.Linear(1, 1)
        
        # Optional activation function
        if config.custom_act is not None:
            self.activation_layer = get_activation_layer(config.custom_act)
        else: 
            self.activation_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear regression model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1)
        """
        # Apply linear transformation
        x = self.linear(x)
        
        # Apply activation function if specified
        return self.activation_layer(x) if self.activation_layer else x 


    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic data for linear regression training.
        
        Creates 100 data points following the linear relationship: y = 100x + noise
        where x is uniformly distributed in [0, 10] and noise is uniform in [0, 1].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (inputs, targets) where:
                - inputs: Random values in range [0, 10] of shape (100, 1)
                - targets: Linear function of inputs with noise of shape (100, 1)
        """
        if random_seed: 
            import random
            import numpy as np 
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Generate random inputs in range [0, 10]
        inputs = torch.rand(100, 1) * 10
        
        # Generate targets with linear relationship y = 100x + noise
        targets = 100 * inputs + torch.rand(100, 1)
        
        return inputs, targets