"""
Copyright (c) 2025. All rights reserved.
"""

"""
Activation function utilities for neural networks.

This module provides a factory function to create various PyTorch activation
layers including ReLU, Tanh, Sigmoid, LeakyReLU, GELU, and SiLU. Used by
regression models to add non-linearity.
"""

import torch 

def get_activation_layer(custom_act: str) -> torch.nn.Module:
    """
    Factory function to create PyTorch activation layer instances.
    
    This function returns the appropriate PyTorch activation layer based on the
    provided string identifier. Supports commonly used activation functions
    for neural network architectures.
    
    Args:
        custom_act (str): Name of the activation function to create.
                         Supported values:
                         - "relu": Rectified Linear Unit
                         - "tanh": Hyperbolic Tangent  
                         - "sigmoid": Sigmoid function
                         - "leakyrelu": Leaky ReLU with default negative slope
                         - "gelu": Gaussian Error Linear Unit
                         - "silu": Sigmoid Linear Unit (Swish)
    
    Returns:
        torch.nn.Module: Initialized PyTorch activation layer
    
    Raises:
        ValueError: If custom_act is not a supported activation function
        
    Example:
        activation = get_activation_layer("relu")  # Returns nn.ReLU()
        activation = get_activation_layer("gelu")  # Returns nn.GELU()
    """
    if custom_act == "relu": 
        return torch.nn.ReLU() 
    elif custom_act == "tanh": 
        return torch.nn.Tanh()
    elif custom_act == "sigmoid":
        return torch.nn.Sigmoid()
    elif custom_act == "leakyrelu":
        return torch.nn.LeakyReLU()
    elif custom_act == "gelu":
        return torch.nn.GELU()
    elif custom_act == "silu": 
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation layer: {custom_act}. "
                        f"Supported activations: relu, tanh, sigmoid, leakyrelu, gelu, silu")