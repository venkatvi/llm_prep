"""
Linear regression model: y = Wx + b with optional activation.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Tuple

import torch

from lib.activations import get_activation_layer
from lib.configs import ModelConfig
from lib.utils import set_seed


class LinearRegressionModel(torch.nn.Module):
    """Linear regression model with optional activation function.

    A simple linear regression model that maps single-dimensional inputs to
    single-dimensional outputs using the equation y = Wx + b, with an optional
    activation function applied to the output.

    Attributes:
        linear (torch.nn.Linear): Linear transformation layer (1 -> 1)
        activation_layer (Optional[torch.nn.Module]): Optional activation function
    """

    linear: torch.nn.Linear
    activation_layer: Optional[torch.nn.Module]

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the linear regression model.

        Args:
            config (ModelConfig): Model configuration containing activation settings
        """
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

        if config.custom_act is not None:
            self.activation_layer = get_activation_layer(config.custom_act)
        else:
            self.activation_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear regression model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1]
        """
        x = self.linear(x)
        return self.activation_layer(x) if self.activation_layer else x

    def generate_data(self, random_seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic linear regression data.

        Creates synthetic data following the linear relationship y = 100x + noise,
        where x is uniformly distributed in [0, 10] and noise is uniform in [0, 1].

        Args:
            random_seed (Optional[int]): Random seed for reproducible data generation

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - inputs: Input tensor of shape [100, 1] with values in [0, 10]
                - targets: Target tensor of shape [100, 1] following y = 100x + noise
        """
        if random_seed:
            set_seed(random_seed)

        # Generate random inputs in range [0, 10]
        inputs: torch.Tensor = torch.rand(100, 1) * 10

        # Generate targets with linear relationship y = 100x + noise
        targets: torch.Tensor = 100 * inputs + torch.rand(100, 1)

        return inputs, targets
