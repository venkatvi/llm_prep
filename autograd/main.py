"""
Demo: Linear layer + LearnedSiLU activation with gradient computation.
"""

import torch

from activations import LearnedSiLU
from linear import Linear 

if __name__ == "__main__": 
    # Initialize parameters
    x: torch.Tensor = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)
    w: torch.Tensor = torch.tensor([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
    b: torch.Tensor = torch.tensor([[-5.0, -12.0]], requires_grad=True)
    slope: torch.Tensor = torch.tensor([-1.0], requires_grad=True)

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
