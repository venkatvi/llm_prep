"""
Linear layer implementation: y = xW^T + b
"""

import torch
from typing import Tuple

class Linear(torch.autograd.Function):
    """Linear transformation: y = xW^T + b"""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input, weights, bias)
        return input @ weights.T + bias
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input, weights, bias = ctx.saved_tensors
        grad_input = grad_output @ weights
        grad_weights = grad_output.T @ input
        grad_bias = grad_output
        return grad_input, grad_weights, grad_bias
