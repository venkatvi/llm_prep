"""
Copyright (c) 2025. All rights reserved.
"""

"""
Activation functions implemented as PyTorch autograd Functions.

This module provides custom implementations of essential neural network activation
functions that introduce non-linearity into neural networks. Each activation function
includes both forward computation and custom gradient calculation, demonstrating
the mathematical principles behind backpropagation.

Activation functions are crucial components that:
- Enable neural networks to learn complex, non-linear mappings
- Control gradient flow during backpropagation
- Determine the output range and behavior of neurons

Classes:
    Tanh: Hyperbolic tangent activation with output range (-1, 1)
    Sigmoid: Logistic sigmoid activation with output range (0, 1)  
    ReLU: Rectified Linear Unit with output range [0, ∞)
    LearnedSiLU: Learnable Sigmoid Linear Unit with trainable scaling

Example:
    >>> import torch
    >>> from activations import Tanh, Sigmoid, ReLU, LearnedSiLU
    
    >>> # Basic activation usage
    >>> x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    >>> 
    >>> tanh_out = Tanh.apply(x)       # [-0.964, 0.0, 0.964]
    >>> sigmoid_out = Sigmoid.apply(x) # [0.119, 0.5, 0.881]
    >>> relu_out = ReLU.apply(x)       # [0.0, 0.0, 2.0]
    >>> 
    >>> # Learnable activation with parameter
    >>> slope = torch.tensor([1.5], requires_grad=True)
    >>> silu_out = LearnedSiLU.apply(x, slope)
    
Mathematical Properties:
    Each activation function has specific mathematical properties:
    
    1. Tanh: f(x) = tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
       - Symmetric around origin, bounded output
       - Derivative: f'(x) = 1 - tanh²(x)
    
    2. Sigmoid: f(x) = 1/(1 + e^(-x))
       - Smooth, monotonic, bounded between 0 and 1
       - Derivative: f'(x) = σ(x)(1 - σ(x))
    
    3. ReLU: f(x) = max(0, x)
       - Simple, computationally efficient, unbounded positive output
       - Derivative: f'(x) = 1 if x > 0, else 0
    
    4. LearnedSiLU: f(x) = α * x * σ(x) where α is learnable
       - Combines linear and sigmoid components with trainable scaling
       - Smooth activation with learnable curvature
"""

import torch 
from simple import Exp
from typing import Tuple

class Tanh(torch.autograd.Function): 
    """
    Custom autograd function for hyperbolic tangent activation.
    
    The hyperbolic tangent is a smooth, differentiable activation function that
    maps input values to the range (-1, 1). It's symmetric around the origin
    and is commonly used in RNNs and traditional neural networks.
    
    Mathematical Formulation:
        Forward: f(x) = tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
        Backward: f'(x) = 1 - tanh²(x) = sech²(x)
    
    Properties:
        - Output range: (-1, 1)
        - Zero-centered output (mean around 0)
        - Smooth and differentiable everywhere
        - Can suffer from vanishing gradients for large |x|
    
    Args:
        input (torch.Tensor): Input tensor to apply tanh activation
    
    Returns:
        torch.Tensor: Tanh-activated output in range (-1, 1)
        
    Example:
        >>> x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        >>> y = Tanh.apply(x)  # [-0.964, 0.0, 0.964]
        >>> loss = y.sum()
        >>> loss.backward()
        >>> print(x.grad)  # Gradients based on 1 - tanh²(x)
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor
    ) -> torch.Tensor: 
        """
        Compute tanh(x) and save input for backward pass.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor to apply tanh activation
            
        Returns:
            torch.Tensor: Tanh-activated output
        """
        ctx.save_for_backward(input)
        output = torch.tanh(input)  # Use torch.tanh for tensor support
        return output 
    
    @staticmethod 
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient using tanh derivative: d/dx(tanh(x)) = 1 - tanh²(x).
        
        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            torch.Tensor: Gradient w.r.t. input using chain rule
        """
        (input, ) = ctx.saved_tensors 
        tanh_val = torch.tanh(input)
        grad_input = grad_output * (1 - tanh_val ** 2)  # Apply chain rule
        return grad_input

class Sigmoid(torch.autograd.Function): 
    """
    Custom autograd function for sigmoid (logistic) activation.
    
    The sigmoid function maps input values to the range (0, 1), making it useful
    for binary classification and probability estimation. It's smooth and differentiable
    but can suffer from vanishing gradients for extreme input values.
    
    Mathematical Formulation:
        Forward: f(x) = σ(x) = 1/(1 + e^(-x))
        Backward: f'(x) = σ(x)(1 - σ(x))
    
    Properties:
        - Output range: (0, 1)
        - Monotonically increasing
        - Smooth S-curve shape
        - Can saturate (gradient approaches 0) for large |x|
        - Commonly used in binary classification output layers
    
    Args:
        input (torch.Tensor): Input tensor to apply sigmoid activation
    
    Returns:
        torch.Tensor: Sigmoid-activated output in range (0, 1)
        
    Example:
        >>> x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        >>> y = Sigmoid.apply(x)  # [0.119, 0.5, 0.881]
        >>> loss = y.sum()
        >>> loss.backward()
        >>> print(x.grad)  # Gradients based on σ(x)(1-σ(x))
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor
    ) -> torch.Tensor: 
        """
        Compute sigmoid(x) = 1/(1 + e^(-x)) and save input for backward pass.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor to apply sigmoid activation
            
        Returns:
            torch.Tensor: Sigmoid-activated output in range (0, 1)
        """
        ctx.save_for_backward(input)
        output = 1 / (1.0 + Exp.apply(-input))  # Using custom Exp function
        return output 
    
    @staticmethod 
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient using sigmoid derivative: d/dx(σ(x)) = σ(x)(1 - σ(x)).
        
        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            torch.Tensor: Gradient w.r.t. input using chain rule
        """
        (input, ) = ctx.saved_tensors 
        sigmoid = 1 / (1.0 + Exp.apply(-input))  # Recompute sigmoid
        grad_input = grad_output * sigmoid * (1 - sigmoid)  # Apply chain rule
        return grad_input 
    
class ReLU(torch.autograd.Function):
    """
    Custom autograd function for Rectified Linear Unit (ReLU) activation.
    
    ReLU is the most popular activation function in modern deep learning due to
    its simplicity and effectiveness. It helps mitigate the vanishing gradient
    problem and is computationally efficient.
    
    Mathematical Formulation:
        Forward: f(x) = max(0, x)
        Backward: f'(x) = 1 if x > 0, else 0
    
    Properties:
        - Output range: [0, ∞)
        - Non-saturating for positive inputs
        - Sparse activation (many neurons output 0)
        - Can suffer from "dying ReLU" problem (neurons stuck at 0)
        - Computationally very efficient
    
    Args:
        input (torch.Tensor): Input tensor to apply ReLU activation
    
    Returns:
        torch.Tensor: ReLU-activated output (negative values zeroed)
        
    Example:
        >>> x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        >>> y = ReLU.apply(x)  # [0.0, 0.0, 2.0]
        >>> loss = y.sum()
        >>> loss.backward()
        >>> print(x.grad)  # [0.0, 0.0, 1.0] (gradient = 1 where x > 0)
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor
    ) -> torch.Tensor: 
        """
        Compute ReLU(x) = max(0, x) and save input for backward pass.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor to apply ReLU activation
            
        Returns:
            torch.Tensor: ReLU-activated output (negative values set to 0)
        """
        ctx.save_for_backward(input)
        output = torch.clamp(input, min=0)  # ReLU: max(0, x)
        return output 
    
    @staticmethod 
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient using ReLU derivative: d/dx(ReLU(x)) = 1 if x > 0, else 0.
        
        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            torch.Tensor: Gradient w.r.t. input (0 for negative inputs, grad_output for positive)
        """
        (input, ) = ctx.saved_tensors 
        grad_input = grad_output * (input > 0).float()  # Apply chain rule
        return grad_input 
    
class LearnedSiLU(torch.autograd.Function):
    """
    Custom autograd function for learnable SiLU (Sigmoid Linear Unit) activation.
    
    This is a parameterized version of the SiLU activation function with a learnable
    scaling parameter. SiLU combines the benefits of sigmoid gating with linear growth,
    and the learnable parameter allows the network to adapt the activation's behavior.
    
    Mathematical Formulation:
        Forward: f(x, α) = α × x × σ(x) where σ is sigmoid and α is learnable
        Backward: 
            - ∂f/∂x = α × [σ(x) + x × σ'(x)] (product rule)
            - ∂f/∂α = x × σ(x) (parameter gradient)
    
    Properties:
        - Smooth, differentiable activation
        - Self-gating mechanism via sigmoid
        - Learnable scaling allows adaptation to different tasks
        - Non-monotonic shape with both positive and negative regions
        - Can help with gradient flow compared to traditional activations
    
    Args:
        input (torch.Tensor): Input tensor to apply activation
        slope (torch.Tensor): Learnable scaling parameter
    
    Returns:
        torch.Tensor: Activated output scaled by learnable parameter
        
    Example:
        >>> x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> alpha = torch.tensor([2.0], requires_grad=True)
        >>> y = LearnedSiLU.apply(x, alpha)  # Scaled SiLU activation
        >>> loss = y.sum()
        >>> loss.backward()
        >>> print(x.grad)     # Input gradients
        >>> print(alpha.grad) # Parameter gradients
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor, 
        slope: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute learnable SiLU: slope × x × sigmoid(x).
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor to apply activation
            slope: Learnable scaling parameter
            
        Returns:
            torch.Tensor: Scaled SiLU-activated output
        """
        ctx.save_for_backward(input, slope)
        sigmoid_x = Sigmoid.apply(input)
        output = slope * input * sigmoid_x
        return output 

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients for learnable SiLU using product rule and chain rule.
        
        The gradient computation involves the product rule for f(x) = α × x × σ(x):
        - Input gradient uses product rule: ∂f/∂x = α × [σ(x) + x × σ'(x)]
        - Parameter gradient: ∂f/∂α = x × σ(x)
        
        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Tuple containing:
                - grad_input: Gradient w.r.t. input tensor
                - grad_slope: Gradient w.r.t. learnable slope parameter
        """
        input, slope = ctx.saved_tensors 
        
        sigmoid_x = Sigmoid.apply(input)
        sigmoid_prime = sigmoid_x * (1 - sigmoid_x)  # σ'(x) = σ(x)(1 - σ(x))
        
        # Gradient w.r.t. input using product rule: ∂f/∂x = α × [σ(x) + x × σ'(x)]
        grad_input = grad_output * slope * (sigmoid_x + input * sigmoid_prime)
        
        # Gradient w.r.t. slope parameter: ∂f/∂α = x × σ(x)
        grad_slope = (grad_output * input * sigmoid_x).sum(dim=0, keepdim=True)  # Preserve shape
        
        return grad_input, grad_slope  
       
        