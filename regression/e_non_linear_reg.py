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
    """Multi-Layer Perceptron for non-linear regression.

    A configurable MLP architecture with multiple hidden layers, activation functions,
    and optional residual connections. Designed for non-linear regression tasks where
    a single linear layer is insufficient.

    Attributes:
        config (ModelConfig): Configuration object containing layer specifications
        layers (torch.nn.ModuleList): List of linear and activation layers
        output_layer (torch.nn.Linear): Final linear layer mapping to output dimension
    """

    config: ModelConfig
    layers: torch.nn.ModuleList
    output_layer: torch.nn.Linear

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the MLP with configurable architecture.

        Args:
            config (ModelConfig): Configuration object containing:
                - num_latent_layers: Number of hidden layers
                - latent_dims: List of hidden layer dimensions
                - custom_act: Activation function name
                - allow_residual: Whether to enable residual connections

        Raises:
            AssertionError: If length of latent_dims doesn't match num_latent_layers
        """
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()

        assert (
            len(self.config.latent_dims) == self.config.num_latent_layers
        ), f"Number of dimensions ({len(self.config.latent_dims)}) must match number of layers ({self.config.num_latent_layers})"

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
        """Forward pass through the MLP.

        Processes input through alternating linear and activation layers.
        When residual connections are enabled, adds input to activation outputs.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1]
        """
        for layer in self.layers:
            if not isinstance(layer, torch.nn.Linear) and self.config.allow_residual:
                x = layer(x) + x
            else:
                x = layer(x)

        return self.output_layer(x)

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic non-linear regression data.

        Creates synthetic data following the quadratic relationship y = 4x² + 2x + noise,
        where x is uniformly distributed in [0, 10] and noise is uniform in [0, 1].
        This non-linear relationship requires a multi-layer network to model effectively.

        Args:
            random_seed (Optional[int]): Random seed for reproducible data generation

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - inputs: Input tensor of shape [100, 1] with values in [0, 10]
                - targets: Target tensor of shape [100, 1] following y = 4x² + 2x + noise
        """
        if random_seed:
            set_seed(random_seed)

        # Generate random inputs in range [0, 10]
        inputs: torch.Tensor = torch.rand(100, 1) * 10

        # Generate targets with quadratic relationship y = 4x² + 2x + noise
        targets: torch.Tensor = 4 * inputs**2 + 2 * inputs + torch.rand(100, 1)

        return inputs, targets
