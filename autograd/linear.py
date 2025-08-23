"""
Linear layer implementation: y = xW^T + b
"""

from typing import Tuple

import torch


class Linear(torch.autograd.Function):
    """Custom implementation of linear transformation layer.

    Implements the linear transformation y = xW^T + b commonly used in
    fully connected neural network layers. This custom autograd function
    demonstrates how to compute gradients with respect to inputs, weights,
    and bias terms.

    The transformation takes input features and applies a learned linear
    mapping followed by a bias addition, which is the foundation of
    most neural network architectures.
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass computing y = xW^T + b.

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input features [batch_size, input_dim]
            weights (torch.Tensor): Weight matrix [output_dim, input_dim]
            bias (torch.Tensor): Bias vector [output_dim] or [batch_size, output_dim]

        Returns:
            torch.Tensor: Linear transformation output [batch_size, output_dim]
        """
        ctx.save_for_backward(input, weights, bias)
        return input @ weights.T + bias

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward pass computing gradients for input, weights, and bias.

        Computes:
        - ∂L/∂x = grad_output @ W
        - ∂L/∂W = grad_output^T @ x
        - ∂L/∂b = grad_output (summed over batch if needed)

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers [batch_size, output_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients with respect to
                (input, weights, bias)
        """
        input, weights, bias = ctx.saved_tensors
        grad_input = grad_output @ weights
        grad_weights = grad_output.T @ input
        grad_bias = grad_output
        return grad_input, grad_weights, grad_bias
