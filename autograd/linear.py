"""
Copyright (c) 2025. All rights reserved.
"""

"""
Linear layer implementation as PyTorch autograd Function.

This module provides a custom implementation of the fundamental building block
of neural networks: the linear (fully connected) layer. It demonstrates matrix
multiplication operations, bias addition, and proper gradient computation using
matrix calculus rules.

The linear layer performs the transformation: y = xW^T + b
where x is the input, W are the learnable weights, and b is the bias term.

Classes:
    Linear: Custom autograd function implementing linear transformation

Example:
    >>> import torch
    >>> from linear import Linear
    
    >>> # Create inputs
    >>> x = torch.randn(3, 4, requires_grad=True)  # [batch_size, input_features]
    >>> w = torch.randn(2, 4, requires_grad=True)  # [output_features, input_features]
    >>> b = torch.randn(3, 2, requires_grad=True)  # [batch_size, output_features]
    
    >>> # Forward pass
    >>> y = Linear.apply(x, w, b)  # Shape: [3, 2]
    >>> loss = y.sum()
    >>> loss.backward()
    
    >>> print(x.grad.shape)  # [3, 4] - same as input
    >>> print(w.grad.shape)  # [2, 4] - same as weights
    >>> print(b.grad.shape)  # [3, 2] - same as bias

Mathematical Background:
    The linear transformation is the foundation of feedforward neural networks.
    Given input x ∈ R^{batch×in}, weights W ∈ R^{out×in}, and bias b ∈ R^{batch×out}:
    
    Forward: y = xW^T + b
    
    Gradients (using chain rule and matrix calculus):
        ∂L/∂x = ∂L/∂y @ W        (gradient flows back through weights)
        ∂L/∂W = ∂L/∂y^T @ x      (outer product of gradients and inputs)
        ∂L/∂b = ∂L/∂y            (bias gradient equals output gradient)
"""

import torch
from typing import Tuple

class Linear(torch.autograd.Function):
    """
    Custom autograd function for linear transformation y = xW^T + b.
    
    This implements a fully connected layer with learnable weights and bias.
    The layer performs matrix multiplication followed by bias addition, which
    is the core operation in neural networks.
    
    Mathematical Formulation:
        Forward: y = xW^T + b
        where:
            - x: input tensor [batch_size, input_features]
            - W: weight matrix [output_features, input_features]  
            - b: bias vector [batch_size, output_features]
            - y: output tensor [batch_size, output_features]
    
    Backward Pass:
        Using matrix calculus and the chain rule:
        - ∂L/∂x = ∂L/∂y @ W        (input gradient)
        - ∂L/∂W = ∂L/∂y^T @ x      (weight gradient)
        - ∂L/∂b = ∂L/∂y            (bias gradient)
    
    Args:
        input (torch.Tensor): Input features [batch_size, input_features]
        weights (torch.Tensor): Weight matrix [output_features, input_features]
        bias (torch.Tensor): Bias term [batch_size, output_features]
    
    Returns:
        torch.Tensor: Linear transformation output [batch_size, output_features]
        
    Example:
        >>> # Single sample with 3 input features, 2 output features
        >>> x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        >>> w = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], requires_grad=True)
        >>> b = torch.tensor([[0.1, 0.2]], requires_grad=True)
        >>> y = Linear.apply(x, w, b)
        >>> print(y)  # Output: tensor([[5.6, 11.2]], grad_fn=...)
        
    Note:
        This implementation uses per-sample bias (bias shape matches batch dimension)
        which differs from PyTorch's standard linear layer that uses shared bias.
    """
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        input: torch.Tensor, 
        weights: torch.Tensor, 
        bias: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute linear transformation y = xW^T + b and save tensors for backward.
        
        This performs the core linear layer computation: matrix multiplication
        of input with transposed weights, followed by bias addition.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input: Input tensor [batch_size, input_features]
            weights: Weight matrix [output_features, input_features] 
            bias: Bias tensor [batch_size, output_features]
            
        Returns:
            torch.Tensor: Linear transformation result [batch_size, output_features]
            
        Mathematical Operation:
            y = xW^T + b
            where @ denotes matrix multiplication and + is element-wise addition
        """
        ctx.save_for_backward(input, weights, bias)  
        # Matrix multiplication: input @ weights.T + bias
        output = input @ weights.T + bias 
        return output 
    
    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, 
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gradients for linear transformation using matrix calculus rules.
        
        This implements the backward pass for the linear layer, computing gradients
        with respect to inputs, weights, and bias using the chain rule and matrix
        differentiation rules.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output: Gradient of loss w.r.t. output [batch_size, output_features]
            
        Returns:
            Tuple containing:
                - grad_input: Gradient w.r.t. input [batch_size, input_features]
                - grad_weights: Gradient w.r.t. weights [output_features, input_features]
                - grad_bias: Gradient w.r.t. bias [batch_size, output_features]
                
        Mathematical Derivation:
            Given y = xW^T + b and ∂L/∂y (grad_output), we compute:
            
            1. ∂L/∂x = ∂L/∂y @ W
               - Chain rule: gradient flows back through weights
               - Shape: [batch, out] @ [out, in] = [batch, in]
               
            2. ∂L/∂W = ∂L/∂y^T @ x  
               - Gradient of matrix multiplication w.r.t. weight matrix
               - Shape: [out, batch] @ [batch, in] = [out, in]
               
            3. ∂L/∂b = ∂L/∂y
               - Bias gradient equals output gradient (∂/∂b(y) = I)
               - Shape: [batch, out]
        """
        input, weights, bias = ctx.saved_tensors 
        
        # Gradient w.r.t. input: ∂L/∂x = ∂L/∂y @ W
        grad_input = grad_output @ weights        # [batch, out] @ [out, in] = [batch, in]
        
        # Gradient w.r.t. weights: ∂L/∂W = ∂L/∂y^T @ x
        grad_weights = grad_output.T @ input      # [out, batch] @ [batch, in] = [out, in]
        
        # Gradient w.r.t. bias: ∂L/∂b = ∂L/∂y (identity derivative)
        grad_bias = grad_output  # Bias gradient has same shape as grad_output
        
        return grad_input, grad_weights, grad_bias
