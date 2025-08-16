"""
Copyright (c) 2025. All rights reserved.
"""

"""
Non-linear regression model implementation.

This module implements a Multi-Layer Perceptron (MLP) for non-linear regression
using PyTorch. The model can learn complex non-linear relationships in data.
"""

import torch
from typing import Optional, Tuple

from activations import get_activation_layer
from configs import ModelConfig
class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) for non-linear regression.
    
    This model implements a feedforward neural network with configurable hidden layers,
    activation functions, and optional residual connections. It can learn complex
    non-linear mappings from input features to target values.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the MLP model with specified architecture.
        
        Args:
            num_latent_layers (int): Number of hidden layers in the network
            latent_dim (list[int]): List of hidden layer dimensions. Length must equal num_latent_layers.
                                   Example: [128, 64, 32] for 3 hidden layers
            custom_act (str): Activation function to use after each hidden layer.
                             Supported: relu, tanh, sigmoid, leakyrelu, gelu, silu
            allow_residual (bool): If True, add residual connections after activation functions.
                                  Default is False.
                                  
        Raises:
            AssertionError: If len(latent_dim) != num_latent_layers
        """
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        
        # Validate that dimensions match layer count
        assert len(self.config.latent_dims) == self.config.num_latent_layers, \
            f"Number of dimensions ({len(self.config.latent_dims)}) must match number of layers ({self.config.num_latent_layers})"
        
        # Build hidden layers with alternating linear and activation layers
        for layer_index in range(self.config.num_latent_layers): 
            # Determine input dimension: 1 for first layer, previous layer's output for subsequent layers
            input_dim = 1 if layer_index == 0 else output_dim
            output_dim = self.config.latent_dims[layer_index]
            
            # Add linear transformation layer
            linear_layer = torch.nn.Linear(input_dim, output_dim)
            self.layers.append(linear_layer)

            # Add activation function layer
            act_layer = get_activation_layer(self.config.custom_act)
            self.layers.append(act_layer)
        
        # Final output layer maps to single regression output
        self.output_layer = torch.nn.Linear(output_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, 1)
        """
        # Pass through all hidden layers (linear + activation pairs)
        for layer in self.layers:
            if not isinstance(layer, torch.nn.Linear) and self.config.allow_residual: 
                # Apply residual connection: activation(x) + x
                x = layer(x) + x
            else:
                # Standard layer application
                x = layer(x)
        
        # Apply final output layer to get regression prediction
        return self.output_layer(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic non-linear data for MLP training.
        
        Creates 100 data points following a quadratic relationship: y = 4x² + 2x + noise
        where x is uniformly distributed in [0, 10] and noise is uniform in [0, 1].
        This non-linear relationship requires multiple layers to learn effectively.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (inputs, targets) where:
                - inputs: Random values in range [0, 10] of shape (100, 1)
                - targets: Quadratic function of inputs with noise of shape (100, 1)
        """
        if random_seed: 
            import random 
            import numpy as np
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Generate random inputs in range [0, 10]
        inputs = torch.rand(100, 1) * 10
        
        # Generate targets with quadratic relationship y = 4x² + 2x + noise
        targets = 4 * inputs**2 + 2 * inputs + torch.rand(100, 1)
        
        return inputs, targets