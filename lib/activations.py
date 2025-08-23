"""
Activation function factory for neural networks.
"""

import torch


def get_activation_layer(custom_act: str) -> torch.nn.Module:
    """Factory for activation layers: relu, tanh, sigmoid, leakyrelu, gelu, silu"""
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
