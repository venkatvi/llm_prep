"""
Copyright (c) 2025. All rights reserved.
"""

"""
Custom PyTorch autograd implementations for educational purposes.

Modules:
    simple: Mathematical functions (Power, Square, Cube, Exp)
    linear: Linear layer implementation
    activations: Activation functions (Tanh, Sigmoid, ReLU, LearnedSiLU)
"""

from .activations import LearnedSiLU, ReLU, Sigmoid, Tanh
from .linear import Linear

# Import main classes for easy access
from .simple import Cube, Exp, Power, Square

__version__ = "1.0.0"
__author__ = "PyTorch Autograd Educational Implementation"

__all__ = [
    # Mathematical functions
    "Power",
    "Square",
    "Cube",
    "Exp",
    # Neural network layers
    "Linear",
    # Activation functions
    "Tanh",
    "Sigmoid",
    "ReLU",
    "LearnedSiLU",
]
