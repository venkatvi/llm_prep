"""
Copyright (c) 2025. All rights reserved.
"""

"""
Activation functions: Tanh, Sigmoid, ReLU, LearnedSiLU
"""

from typing import Tuple

import torch
from simple import Exp


class Tanh(torch.autograd.Function):
    """Custom implementation of hyperbolic tangent activation function.

    Implements tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) with custom gradient computation.
    The gradient is computed as d/dx tanh(x) = 1 - tanh²(x), which is efficient
    since we can reuse the forward pass result.

    This implementation demonstrates how to create custom autograd functions
    with proper gradient computation for neural network training.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Forward pass computing tanh(x).

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor of any shape

        Returns:
            torch.Tensor: tanh(input) with same shape as input, values in [-1, 1]
        """
        ctx.save_for_backward(input)
        return torch.tanh(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass computing gradient using tanh derivative.

        Uses the identity: d/dx tanh(x) = 1 - tanh²(x)

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers

        Returns:
            torch.Tensor: Gradient with respect to input
        """
        (input,) = ctx.saved_tensors
        tanh_val = torch.tanh(input)
        return grad_output * (1 - tanh_val**2)


class Sigmoid(torch.autograd.Function):
    """Custom implementation of sigmoid activation function.

    Implements σ(x) = 1/(1 + e^(-x)) with custom gradient computation.
    The gradient is computed as d/dx σ(x) = σ(x)(1 - σ(x)), which is efficient
    since we can reuse the forward pass result.

    Uses the custom Exp function for the exponential computation to maintain
    consistency with the custom autograd implementation framework.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Forward pass computing sigmoid(x).

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor of any shape

        Returns:
            torch.Tensor: sigmoid(input) with same shape as input, values in (0, 1)
        """
        ctx.save_for_backward(input)
        return 1 / (1.0 + Exp.apply(-input))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass computing gradient using sigmoid derivative.

        Uses the identity: d/dx σ(x) = σ(x)(1 - σ(x))

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers

        Returns:
            torch.Tensor: Gradient with respect to input
        """
        (input,) = ctx.saved_tensors
        sigmoid = 1 / (1.0 + Exp.apply(-input))
        return grad_output * sigmoid * (1 - sigmoid)


class ReLU(torch.autograd.Function):
    """Custom implementation of Rectified Linear Unit (ReLU) activation function.

    Implements ReLU(x) = max(0, x) with custom gradient computation.
    The gradient is 1 for positive inputs and 0 for negative inputs.
    This creates a piecewise linear function that helps with gradient flow
    in deep neural networks.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Forward pass computing ReLU(x) = max(0, x).

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor of any shape

        Returns:
            torch.Tensor: ReLU(input) with same shape as input, values in [0, ∞)
        """
        ctx.save_for_backward(input)
        return torch.clamp(input, min=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass computing gradient using ReLU derivative.

        The gradient is 1 where input > 0 and 0 elsewhere.

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers

        Returns:
            torch.Tensor: Gradient with respect to input (0 or 1 elementwise)
        """
        (input,) = ctx.saved_tensors
        return grad_output * (input > 0).float()


class LearnedSiLU(torch.autograd.Function):
    """Custom implementation of learnable Sigmoid Linear Unit (SiLU) activation.

    Implements a parameterized version of SiLU: f(x) = α × x × σ(x)
    where α is a learnable parameter and σ(x) is the sigmoid function.

    This activation function combines the smoothness of sigmoid with the
    unbounded positive range, while allowing the network to learn the
    optimal scaling factor α during training.

    The function computes gradients with respect to both the input x
    and the learnable parameter α.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
        """Forward pass computing α × x × σ(x).

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor of any shape
            slope (torch.Tensor): Learnable scaling parameter α

        Returns:
            torch.Tensor: Scaled SiLU output with same shape as input
        """
        ctx.save_for_backward(input, slope)
        sigmoid_x = Sigmoid.apply(input)
        return slope * input * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass computing gradients for both input and slope parameter.

        Computes:
        - ∂f/∂x = α × (σ(x) + x × σ'(x)) where σ'(x) = σ(x)(1 - σ(x))
        - ∂f/∂α = x × σ(x) (summed across batch dimension)

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradients with respect to input and slope
        """
        input, slope = ctx.saved_tensors

        sigmoid_x = Sigmoid.apply(input)
        sigmoid_prime = sigmoid_x * (1 - sigmoid_x)

        grad_input = grad_output * slope * (sigmoid_x + input * sigmoid_prime)
        grad_slope = (grad_output * input * sigmoid_x).sum(dim=0, keepdim=True)

        return grad_input, grad_slope
