"""
Mathematical functions with custom autograd implementation.

Classes: Power, Square, Cube, Exp
"""

from typing import Tuple

import torch


class Power(torch.autograd.Function):
    """Custom implementation of power function x^n with dual gradients.

    Computes x^n where both x and n can require gradients, demonstrating
    how to handle functions with multiple differentiable parameters.
    This is useful for implementing learnable exponents in neural networks.

    The function computes gradients with respect to both the base (x)
    and the exponent (n), using the standard calculus derivatives.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Forward pass computing x^n.

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Base tensor x
            n (torch.Tensor): Exponent tensor n

        Returns:
            torch.Tensor: Result of x^n elementwise
        """
        ctx.save_for_backward(input, n)
        return input**n

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass computing gradients for both base and exponent.

        Computes:
        - ∂/∂x (x^n) = n * x^(n-1)
        - ∂/∂n (x^n) = x^n * ln(x)

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradients with respect to (input, exponent)
        """
        input, n = ctx.saved_tensors
        grad_input = grad_output * n * input ** (n - 1)
        grad_n = grad_output * input**n * torch.log(input)
        return grad_input, grad_n


class Square(torch.autograd.Function):
    """Custom implementation of square function x^2.

    A specialized version of the power function for the common case of x^2.
    This demonstrates optimized implementations for specific mathematical operations
    where the derivative has a simple, efficient form.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Forward pass computing x^2.

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor x

        Returns:
            torch.Tensor: Element-wise square of input
        """
        ctx.save_for_backward(input)
        return input**2

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass computing gradient 2x.

        Uses the derivative: d/dx (x^2) = 2x

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers

        Returns:
            torch.Tensor: Gradient with respect to input
        """
        (input,) = ctx.saved_tensors
        return grad_output * 2 * input


class Cube(torch.autograd.Function):
    """Custom implementation of cube function x^3.

    Another specialized version of the power function for x^3.
    Demonstrates how polynomial operations can be implemented efficiently
    with their well-known derivatives.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Forward pass computing x^3.

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor x

        Returns:
            torch.Tensor: Element-wise cube of input
        """
        ctx.save_for_backward(input)
        return input**3

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass computing gradient 3x^2.

        Uses the derivative: d/dx (x^3) = 3x^2

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers

        Returns:
            torch.Tensor: Gradient with respect to input
        """
        (input,) = ctx.saved_tensors
        return grad_output * 3 * input**2


class Exp(torch.autograd.Function):
    """Custom implementation of exponential function e^x.

    Implements the natural exponential function with its unique property
    that the derivative equals the function itself. This is a fundamental
    building block for many activation functions and probability distributions.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Forward pass computing e^x.

        Args:
            ctx: Context object to save tensors for backward pass
            input (torch.Tensor): Input tensor x

        Returns:
            torch.Tensor: Element-wise exponential of input
        """
        output = torch.exp(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass computing gradient e^x.

        Uses the unique property of exponential: d/dx (e^x) = e^x
        We can reuse the forward pass output for efficiency.

        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream layers

        Returns:
            torch.Tensor: Gradient with respect to input
        """
        (output,) = ctx.saved_tensors
        return grad_output * output
