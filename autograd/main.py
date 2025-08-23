"""
Demonstration of custom autograd functions with gradient computation.

This module demonstrates how to use custom PyTorch autograd functions to build
a simple neural network layer with a linear transformation followed by a
custom learnable activation function. It shows forward pass computation,
gradient computation through backpropagation, and parameter updates.

The example combines:
- Linear layer: y = xW^T + b
- LearnedSiLU activation: z = α × y × σ(y) where α is learnable

All gradients are computed using custom backward passes to demonstrate
the underlying mechanics of automatic differentiation in deep learning frameworks.
"""

import torch
from activations import LearnedSiLU
from linear import Linear

if __name__ == "__main__":
    """
    Main execution demonstrating custom autograd functions.

    This example shows:
    1. Parameter initialization with gradient tracking
    2. Forward pass through linear layer and custom activation
    3. Loss computation and backpropagation
    4. Gradient inspection for all parameters

    The network architecture is: Input -> Linear -> LearnedSiLU -> Sum (loss)
    """
    # Initialize parameters with gradient tracking enabled
    x: torch.Tensor = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)  # Input [1, 3]
    w: torch.Tensor = torch.tensor(
        [[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True
    )  # Weights [2, 3]
    b: torch.Tensor = torch.tensor([[-5.0, -12.0]], requires_grad=True)  # Bias [1, 2]
    slope: torch.Tensor = torch.tensor([-1.0], requires_grad=True)  # Learnable activation parameter

    # Forward pass
    y: torch.Tensor = Linear.apply(x, w, b)
    z: torch.Tensor = LearnedSiLU.apply(y, slope)
    loss: torch.Tensor = z.sum()

    print(f"Linear output: {y}")
    print(f"SiLU output: {z}")
    print(f"Loss: {loss}")

    # Backward pass
    y.retain_grad()
    z.retain_grad()
    loss.backward()

    # Print gradients
    print(f"x.grad: {x.grad}")
    print(f"w.grad: {w.grad}")
    print(f"b.grad: {b.grad}")
    print(f"slope.grad: {slope.grad}")
