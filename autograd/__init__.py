"""
Copyright (c) 2025. All rights reserved.
"""

"""
Custom PyTorch Autograd Implementation Module.

This package provides educational implementations of fundamental PyTorch autograd
functions including mathematical operations, neural network layers, and activation
functions. Each implementation includes both forward computation and custom backward
gradient calculation for learning automatic differentiation concepts.

Example:
    >>> import torch
    >>> from autograd.simple import Square, Cube
    >>> from autograd.linear import Linear
    >>> from autograd.activations import ReLU
    
    >>> # Basic mathematical functions
    >>> x = torch.tensor(3.0, requires_grad=True)
    >>> y = Square.apply(x)  # y = 9.0
    >>> y.backward()
    >>> print(x.grad)  # 6.0 (derivative of x^2 = 2x)
    
    >>> # Neural network components
    >>> x = torch.randn(1, 3, requires_grad=True)
    >>> w = torch.randn(2, 3, requires_grad=True)
    >>> b = torch.randn(1, 2, requires_grad=True)
    >>> 
    >>> linear_out = Linear.apply(x, w, b)
    >>> activated = ReLU.apply(linear_out)

Modules:
    simple: Basic mathematical functions (Power, Square, Cube, Exp)
    linear: Neural network linear layer implementation
    activations: Activation functions (Tanh, Sigmoid, ReLU, LearnedSiLU)
    main: Demonstration script showing integration of components
    tests: Comprehensive test suite for all functions
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