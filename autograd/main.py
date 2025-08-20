"""
Copyright (c) 2025. All rights reserved.
"""

"""
Demonstration of custom autograd functions in a neural network.

This script shows how to combine custom linear layers and activation functions
to create a simple neural network with learnable parameters. Demonstrates
gradient flow through matrix multiplication and custom activation functions.
"""

import torch 
from activations import LearnedSiLU
from linear import Linear 

if __name__ == "__main__": 
    # Create input tensor: [batch_size=1, input_features=3]
    x = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)
    
    # Create weight matrix: [output_features=2, input_features=3]
    # Will be transposed in linear layer for multiplication
    w = torch.tensor([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
    
    # Create bias vector: [batch_size=1, output_features=2]
    b = torch.tensor([[-5.0, -12.0]], requires_grad=True)
    
    # Learnable slope parameter for SiLU activation
    slope = torch.tensor([-1.0], requires_grad=True)

    # Forward pass through custom linear layer: y = xW^T + b
    y = Linear.apply(x, w, b)
    
    # Forward pass through custom learnable SiLU: z = slope * y * sigmoid(y)
    z = LearnedSiLU.apply(y, slope)

    print(f"Linear output y: {y}")
    print(f"LearnedSiLU output z: {z}")

    # Create scalar loss for backward pass (sum all outputs)
    loss = z.sum()  # Convert to scalar for backward()
    print(f"Loss: {loss}")

    # Retain gradients for intermediate tensors (for debugging)
    y.retain_grad()
    z.retain_grad()
    loss.retain_grad() 
    
    # Backward pass: compute all gradients
    loss.backward()

    # Print gradients for analysis
    print(f"z.grad: {z.grad}")
    print(f"y.grad: {y.grad}")
    print(f"x.grad: {x.grad}")
    print(f"w.grad: {w.grad}")
    print(f"b.grad: {b.grad}")
    print(f"slope.grad: {slope.grad}")
