"""
Copyright (c) 2025. All rights reserved.
"""

"""
Activation functions implemented as PyTorch autograd Functions.

This module provides custom implementations of common neural network activation
functions including Tanh, Sigmoid, ReLU, and a learnable SiLU variant.
Each function demonstrates proper gradient computation for backpropagation.
"""

import torch 
import math 
from simple import Exp

class Tanh(torch.autograd.Function): 
    """
    Custom autograd function for hyperbolic tangent activation.
    
    Forward: f(x) = tanh(x)
    Backward: df/dx = 1 - tanh^2(x) = sech^2(x)
    """
    @staticmethod
    def forward(ctx, input): 
        """Compute tanh(x) and save input for backward pass."""
        ctx.save_for_backward(input)
        output = math.tanh(input)
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        """Compute gradient: d/dx(tanh(x)) = 1 - tanh^2(x)."""
        (input, ) = ctx.saved_tensors 
        self_grad = 1 - math.tanh(input) ** 2  # Derivative of tanh
        grad_input = self_grad * grad_output  # Chain rule
        return grad_input

class Sigmoid(torch.autograd.Function): 
    """
    Custom autograd function for sigmoid activation.
    
    Forward: f(x) = 1 / (1 + e^(-x))
    Backward: df/dx = sigmoid(x) * (1 - sigmoid(x))
    """
    @staticmethod
    def forward(ctx, input): 
        """Compute sigmoid(x) = 1/(1 + e^(-x)) and save input for backward pass."""
        ctx.save_for_backward(input)
        output = 1 / (1.0 + Exp.apply(-input))  # Using custom Exp function
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        """Compute gradient: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))."""
        (input, ) = ctx.saved_tensors 
        sigmoid = 1 / (1.0 + Exp.apply(-input))  # Recompute sigmoid
        self_grad = sigmoid * (1 - sigmoid)  # Derivative of sigmoid
        grad_input = self_grad * grad_output  # Chain rule
        return grad_input 
    
class ReLU(torch.autograd.Function):
    """
    Custom autograd function for Rectified Linear Unit activation.
    
    Forward: f(x) = max(0, x)
    Backward: df/dx = 1 if x > 0, else 0
    """
    @staticmethod
    def forward(ctx, input): 
        """Compute ReLU(x) = max(0, x) and save input for backward pass."""
        ctx.save_for_backward(input)
        output = torch.clamp(input, min=0)  # ReLU: max(0, x)
        return output 
    
    @staticmethod 
    def backward(ctx, grad_output):
        """Compute gradient: d/dx(ReLU(x)) = 1 if x > 0, else 0."""
        (input, ) = ctx.saved_tensors 
        self_grad = (input > 0).float()  # 1 if x > 0, else 0
        grad_input = self_grad * grad_output  # Chain rule
        return grad_input 
    
class LearnedSiLU(torch.autograd.Function):
    """
    Custom autograd function for learnable SiLU activation.
    
    Forward: f(x) = slope × x × sigmoid(x)
    Backward: 
        - ∂f/∂x = slope × [sigmoid(x) + x × sigmoid'(x)]
        - ∂f/∂slope = x × sigmoid(x)
    """
    @staticmethod
    def forward(ctx, input, slope):
        """Compute slope * x * sigmoid(x) and save inputs for backward pass."""
        ctx.save_for_backward(input, slope)
        output = slope * input * Sigmoid.apply(input)
        return output 

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients for learnable SiLU using product rule.
        
        Returns:
            grad_input: gradient w.r.t. input x
            grad_slope: gradient w.r.t. learnable slope parameter
        """
        input, slope = ctx.saved_tensors 
        
        sigmoid_x = Sigmoid.apply(input)
        sigmoid_prime = sigmoid_x * (1 - sigmoid_x)  # Sigmoid derivative
        
        # ∂y/∂x = slope × [sigmoid(x) + x × sigmoid'(x)] using product rule
        self_grad_input = grad_output * slope * (sigmoid_x + input * sigmoid_prime)
        
        # ∂y/∂slope = x × sigmoid(x)
        self_grad_slope = grad_output * input * sigmoid_x
        
        return self_grad_input, self_grad_slope  
       
        