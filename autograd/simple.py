"""
Copyright (c) 2025. All rights reserved.
"""

"""
Simple mathematical functions implemented as PyTorch autograd Functions.

This module provides basic mathematical operations (square, cube, exponential)
with custom forward and backward implementations for educational purposes.
Each function demonstrates proper gradient computation using the chain rule.
"""

import torch 
import math

class Square(torch.autograd.Function): 
    """
    Custom autograd function for computing x^2.
    
    Forward: f(x) = x^2
    Backward: df/dx = 2x
    """
    @staticmethod
    def forward(ctx, input):
        """Compute x^2 and save input for backward pass."""
        ctx.save_for_backward(input)
        output = input ** 2 
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output): 
        """Compute gradient: d/dx(x^2) = 2x."""
        (input, ) = ctx.saved_tensors
        self_grad = 2 * input  # Derivative of x^2
        grad_input = grad_output * self_grad  # Chain rule
        return grad_input

class Cube(torch.autograd.Function): 
    """
    Custom autograd function for computing x^3.
    
    Forward: f(x) = x^3
    Backward: df/dx = 3x^2
    """
    @staticmethod
    def forward(ctx, input):
        """Compute x^3 and save input for backward pass."""
        ctx.save_for_backward(input)
        output = input ** 3
        return output 
    
    @staticmethod
    def backward(ctx, grad_output): 
        """Compute gradient: d/dx(x^3) = 3x^2."""
        (input, ) = ctx.saved_tensors
        self_grad = 3 * input ** 2  # Derivative of x^3
        grad_input = grad_output * self_grad  # Chain rule
        return grad_input
    
class Exp(torch.autograd.Function): 
    """
    Custom autograd function for computing e^x.
    
    Forward: f(x) = e^x
    Backward: df/dx = e^x
    """
    @staticmethod
    def forward(ctx, input): 
        """Compute e^x and save input for backward pass."""
        ctx.save_for_backward(input)
        output = torch.exp(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradient: d/dx(e^x) = e^x."""
        (input, ) = ctx.saved_tensors
        self_grad = torch.exp(input)  # Derivative of e^x is e^x
        grad_input = self_grad * grad_output  # Chain rule
        return grad_input