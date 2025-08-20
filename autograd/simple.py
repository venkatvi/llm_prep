"""
Copyright (c) 2025. All rights reserved.
"""

"""
Simple mathematical functions implemented as PyTorch autograd Functions.

This module provides basic mathematical operations with custom forward and backward
implementations for educational purposes. Each function demonstrates proper gradient
computation using the chain rule and serves as a foundation for understanding
automatic differentiation.

Classes:
    Power: Computes x^n with gradients for both base and exponent
    Square: Computes x^2 with derivative 2x
    Cube: Computes x^3 with derivative 3x^2
    Exp: Computes e^x with derivative e^x

Example:
    >>> import torch
    >>> from simple import Square, Power
    
    >>> # Basic usage
    >>> x = torch.tensor(3.0, requires_grad=True)
    >>> y = Square.apply(x)  # y = 9.0
    >>> y.backward()
    >>> print(x.grad)  # 6.0 (derivative: 2 * 3)
    
    >>> # Power function with learnable exponent
    >>> base = torch.tensor(2.0, requires_grad=True)
    >>> exp = torch.tensor(3.0, requires_grad=True)
    >>> result = Power.apply(base, exp)  # 2^3 = 8
    >>> result.backward()
    >>> print(base.grad)  # 12.0 (3 * 2^2)
    >>> print(exp.grad)   # 5.545 (8 * ln(2))

Note:
    All functions support both scalar and tensor inputs, with proper
    broadcasting and gradient computation for educational purposes.
"""

from typing import Tuple
import torch 

class Power(torch.autograd.Function):
    """
    Custom autograd function for computing x^n with gradients for both inputs.
    
    This function demonstrates the product rule and logarithmic differentiation
    for functions of the form f(x,n) = x^n where both base and exponent can
    have gradients.
    
    Mathematical Formulation:
        Forward: f(x,n) = x^n
        Backward: 
            - ∂f/∂x = n * x^(n-1)  (power rule)
            - ∂f/∂n = x^n * ln(x)  (logarithmic differentiation)
    
    Args:
        input (torch.Tensor): Base tensor (x)
        n (torch.Tensor): Exponent tensor (n)
    
    Returns:
        torch.Tensor: Result of x^n
        
    Example:
        >>> x = torch.tensor(2.0, requires_grad=True)
        >>> n = torch.tensor(3.0, requires_grad=True)
        >>> y = Power.apply(x, n)  # y = 8.0
        >>> y.backward()
        >>> print(x.grad)  # 12.0 (3 * 2^2)
        >>> print(n.grad)  # 5.545 (8 * ln(2))
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor, 
        n: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute x^n and save inputs for backward pass.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Base tensor (x)
            n: Exponent tensor (n)
            
        Returns:
            torch.Tensor: x^n
        """
        ctx.save_for_backward(input, n)
        output = input ** n
        return output
    
    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients for power function using differentiation rules.
        
        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Tuple containing:
                - grad_input: Gradient w.r.t. base (∂f/∂x = n * x^(n-1))
                - grad_n: Gradient w.r.t. exponent (∂f/∂n = x^n * ln(x))
        """
        input, n = ctx.saved_tensors
        
        # Gradient w.r.t. input using power rule: ∂/∂x(x^n) = n * x^(n-1)
        grad_input = grad_output * n * input ** (n - 1)
        
        # Gradient w.r.t. exponent using logarithmic differentiation: ∂/∂n(x^n) = x^n * ln(x)
        grad_n = grad_output * input ** n * torch.log(input)
        
        return grad_input, grad_n
        

class Square(torch.autograd.Function):
    """
    Custom autograd function for computing x^2 (square function).
    
    This is a fundamental polynomial function demonstrating the power rule
    for differentiation. The square function is commonly used in loss
    functions and mathematical operations.
    
    Mathematical Formulation:
        Forward: f(x) = x^2
        Backward: df/dx = 2x (power rule)
    
    Args:
        input (torch.Tensor): Input tensor to be squared
    
    Returns:
        torch.Tensor: Element-wise square of input
        
    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = Square.apply(x)  # [1.0, 4.0, 9.0]
        >>> loss = y.sum()
        >>> loss.backward()
        >>> print(x.grad)  # [2.0, 4.0, 6.0] (derivatives: 2*x)
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute x^2 element-wise and save input for backward pass.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor to square
            
        Returns:
            torch.Tensor: Element-wise square of input
        """
        ctx.save_for_backward(input)
        output = input ** 2
        return output
    
    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient using power rule: d/dx(x^2) = 2x.
        
        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            torch.Tensor: Gradient w.r.t. input (2x * grad_output)
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output * 2 * input  # Apply chain rule
        return grad_input

class Cube(torch.autograd.Function): 
    """
    Custom autograd function for computing x^3 (cube function).
    
    This function demonstrates the power rule for cubic polynomials and is
    commonly used in mathematical operations and polynomial approximations.
    
    Mathematical Formulation:
        Forward: f(x) = x^3
        Backward: df/dx = 3x^2 (power rule)
    
    Args:
        input (torch.Tensor): Input tensor to be cubed
    
    Returns:
        torch.Tensor: Element-wise cube of input
        
    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = Cube.apply(x)  # [1.0, 8.0, 27.0]
        >>> loss = y.sum()
        >>> loss.backward()
        >>> print(x.grad)  # [3.0, 12.0, 27.0] (derivatives: 3*x^2)
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute x^3 element-wise and save input for backward pass.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor to cube
            
        Returns:
            torch.Tensor: Element-wise cube of input
        """
        ctx.save_for_backward(input)
        output = input ** 3
        return output 
    
    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> torch.Tensor: 
        """
        Compute gradient using power rule: d/dx(x^3) = 3x^2.
        
        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            torch.Tensor: Gradient w.r.t. input (3x^2 * grad_output)
        """
        (input, ) = ctx.saved_tensors
        grad_input = grad_output * 3 * input ** 2  # Apply chain rule
        return grad_input
    
class Exp(torch.autograd.Function): 
    """
    Custom autograd function for computing e^x (exponential function).
    
    The exponential function is fundamental in mathematics and machine learning,
    particularly in activation functions, probability distributions, and optimization.
    It has the unique property that its derivative equals itself.
    
    Mathematical Formulation:
        Forward: f(x) = e^x
        Backward: df/dx = e^x (self-derivative property)
    
    Args:
        input (torch.Tensor): Input tensor for exponential computation
    
    Returns:
        torch.Tensor: Element-wise exponential of input
        
    Example:
        >>> x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
        >>> y = Exp.apply(x)  # [1.0, 2.718, 7.389]
        >>> loss = y.sum()
        >>> loss.backward()
        >>> print(x.grad)  # [1.0, 2.718, 7.389] (same as output)
        
    Note:
        The exponential function grows rapidly and may cause numerical overflow
        for large input values. Consider using torch.nn.functional.softmax for
        normalized exponentials in neural networks.
    """
    
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor
    ) -> torch.Tensor: 
        """
        Compute e^x element-wise and save result for backward pass.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor for exponential computation
            
        Returns:
            torch.Tensor: Element-wise exponential of input
        """
        output = torch.exp(input)
        ctx.save_for_backward(output)  # Save output since d/dx(e^x) = e^x
        return output
    
    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient using exponential derivative: d/dx(e^x) = e^x.
        
        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            torch.Tensor: Gradient w.r.t. input (e^x * grad_output)
        """
        (output, ) = ctx.saved_tensors  # We saved e^x in forward
        grad_input = grad_output * output  # Apply chain rule
        return grad_input