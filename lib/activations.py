"""
Copyright (c) 2025. All rights reserved.
"""

"""
Activation function factory for neural networks.

This module provides a centralized factory function for creating various
PyTorch activation functions commonly used in deep learning. It supports
both classic activation functions (ReLU, Tanh, Sigmoid) and modern variants
(GELU, SiLU) that have proven effective in recent architectures.

The factory pattern allows for dynamic activation function selection,
making it easy to experiment with different activations during model
development and hyperparameter tuning.

Functions:
    get_activation_layer: Factory for creating activation function instances
"""

import torch


def get_activation_layer(custom_act: str) -> torch.nn.Module:
    """Factory function for creating activation layer instances.

    Creates and returns the appropriate PyTorch activation function module
    based on the string identifier. All activation functions are instantiated
    with default parameters and ready for use in neural network architectures.

    Args:
        custom_act (str): Activation function identifier. Supported values:
                         - 'relu': Rectified Linear Unit
                         - 'tanh': Hyperbolic tangent
                         - 'sigmoid': Sigmoid function
                         - 'leakyrelu': Leaky ReLU with default negative slope
                         - 'gelu': Gaussian Error Linear Unit
                         - 'silu': Sigmoid Linear Unit (also known as Swish)

    Returns:
        torch.nn.Module: Instantiated activation layer ready for use

    Raises:
        ValueError: If custom_act is not one of the supported activation types

    Example:
        activation = get_activation_layer('relu')
        output = activation(input_tensor)
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
        raise ValueError(
            f"Unsupported activation layer: {custom_act}. "
            f"Supported activations: relu, tanh, sigmoid, leakyrelu, gelu, silu"
        )
