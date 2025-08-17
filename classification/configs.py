"""
Copyright (c) 2025. All rights reserved.
"""

"""
Configuration classes for CIFAR-10 classification models.

This module extends the base ModelConfig to include CIFAR-10 specific
parameters such as input channel dimensions for convolutional neural networks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.configs import ModelConfig
from dataclasses import dataclass 

@dataclass
class CIFARModelConfig(ModelConfig):
    """
    Configuration for CIFAR-10 classification models.
    
    Extends the base ModelConfig to include image-specific parameters
    required for convolutional neural networks processing CIFAR-10 data.
    
    Attributes:
        input_channels (int): Number of input color channels (3 for RGB CIFAR-10 images)
        custom_act (str): Activation function name (inherited from ModelConfig)
        num_latent_layers (int): Number of hidden layers (inherited from ModelConfig)
        latent_dims (list[int]): Hidden layer dimensions (inherited from ModelConfig)
        allow_residual (bool): Enable residual connections (inherited from ModelConfig)
    
    Example:
        config = CIFARModelConfig(
            input_channels=3,
            custom_act="relu",
            num_latent_layers=2,
            latent_dims=[128, 64],
            allow_residual=False
        )
    """
    input_channels: int