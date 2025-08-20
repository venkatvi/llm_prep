"""
Copyright (c) 2025. All rights reserved.
"""

"""
Linear layer implementation as PyTorch autograd Function.

This module provides a custom linear transformation y = xW^T + b with 
proper gradient computation for matrix multiplication operations.
Demonstrates how to implement neural network layers from scratch.
"""

import torch

class Linear(torch.autograd.Function):
    """
    Custom autograd function for linear transformation y = xW^T + b.
    
    Forward: y = xW^T + b
    Backward: 
        - ∂L/∂x = ∂L/∂y @ W
        - ∂L/∂W = ∂L/∂y^T @ x  
        - ∂L/∂b = sum(∂L/∂y, dim=0)
    """
    @staticmethod
    def forward(ctx, input, weights, bias):
        """
        Compute linear transformation y = xW^T + b.
        
        Args:
            input: [batch_size, input_features]
            weights: [output_features, input_features] 
            bias: [batch_size, output_features]
        """
        ctx.save_for_backward(input, weights, bias)  
        # Matrix multiplication: input @ weights.T + bias
        output = input @ weights.T + bias 
        return output 
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute gradients for linear transformation using matrix calculus.
        
        Args:
            grad_output: [batch_size, output_features]
            
        Returns:
            grad_input, grad_weights, grad_bias
        """
        input, weights, bias = ctx.saved_tensors 
        
        # Gradients for matrix multiplication
        grad_input = grad_output @ weights        # [batch, out] @ [out, in] = [batch, in]
        grad_weights = grad_output.T @ input      # [out, batch] @ [batch, in] = [out, in]
        grad_bias = grad_output.sum(dim=0, keepdim=True)  # Sum over batch dimension
        
        return grad_input, grad_weights, grad_bias
