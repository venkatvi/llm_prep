import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.configs import ModelConfig
from dataclasses import dataclass 

@dataclass 
class TransformerModelConfig(ModelConfig): 
    input_dim: int 
    embed_dim: int 
    ffn_latent_dim: int 
    num_layers: int 
    output_dim: int 
    num_heads: int 

@dataclass 
class RegressionModelConfig(ModelConfig):
    """
    Configuration for neural network model architecture.
    
    Defines the structure and components of the regression model including
    activation functions, layer dimensions, and architectural features like
    residual connections. Used to customize model complexity and capacity.
    
    Attributes:
        custom_act (str): Activation function ("relu", "tanh", "sigmoid", "leakyrelu", "gelu", "silu")
        num_latent_layers (int): Number of hidden layers in the network
        latent_dims (list[int]): Dimensions for each hidden layer
        allow_residual (bool): Whether to enable residual connections
        
    Example:
        model_config = ModelConfig(
            custom_act="relu",
            num_latent_layers=3,
            latent_dims=[128, 64, 32],
            allow_residual=True
        )
    """
    custom_act: str            # Activation function type
    num_latent_layers: int     # Number of hidden layers
    latent_dims: list[int]     # Hidden layer dimensions
    allow_residual: bool       # Enable residual connections


