"""
Custom PyTorch autograd implementations for educational purposes.

Modules:
    simple: Mathematical functions (Power, Square, Cube, Exp)
    linear: Linear layer implementation
    activations: Activation functions (Tanh, Sigmoid, ReLU, LearnedSiLU)
"""

# Import main classes for easy access
from .simple import Power, Square, Cube, Exp
from .linear import Linear
from .activations import Tanh, Sigmoid, ReLU, LearnedSiLU

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