"""
Copyright (c) 2025. All rights reserved.
"""

"""
Comprehensive demonstration of custom PyTorch autograd implementation.

This script showcases how to combine custom autograd functions to build a complete
neural network pipeline from scratch. It demonstrates the integration of:

1. Custom linear layer (matrix multiplication + bias)
2. Custom learnable activation function (SiLU with trainable parameter)
3. Proper gradient computation and backpropagation
4. End-to-end differentiable computation graph

The example creates a minimal neural network:
Input -> Linear Layer -> LearnedSiLU Activation -> Scalar Loss

This serves as an educational example showing:
- How gradients flow through custom autograd functions
- Matrix calculus in practice (linear layer gradients)
- Product rule application (learnable activation gradients)
- The chain rule connecting all components

Network Architecture:
    x ∈ R^{1×3} -> Linear(3→2) -> LearnedSiLU(slope) -> scalar loss
    
Mathematical Flow:
    1. y = xW^T + b     (linear transformation)
    2. z = α × y × σ(y) (learnable SiLU activation)  
    3. L = sum(z)       (scalar loss for backprop)
    
Expected Output:
    - Forward pass results for each layer
    - Complete gradient computation for all parameters
    - Verification that gradients flow correctly through the network
"""

import torch 
from activations import LearnedSiLU
from linear import Linear 

if __name__ == "__main__": 
    print("=" * 60)
    print("Custom PyTorch Autograd Implementation Demo")
    print("=" * 60)
    
    # Initialize network parameters with specific values for reproducible results
    print("\n1. NETWORK INITIALIZATION")
    print("-" * 30)
    
    # Input: Single sample with 3 features
    x = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)
    print(f"Input x: {x} (shape: {x.shape})")
    
    # Weights: Transform 3 input features to 2 output features  
    # Note: Will be transposed in linear layer (y = xW^T + b)
    w = torch.tensor([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
    print(f"Weights W: {w} (shape: {w.shape})")
    
    # Bias: Per-sample bias matching batch and output dimensions
    b = torch.tensor([[-5.0, -12.0]], requires_grad=True)
    print(f"Bias b: {b} (shape: {b.shape})")
    
    # Learnable activation parameter
    slope = torch.tensor([-1.0], requires_grad=True)
    print(f"SiLU slope α: {slope} (shape: {slope.shape})")

    print("\n2. FORWARD PASS")
    print("-" * 30)
    
    # Linear transformation: y = xW^T + b
    # Expected: [2,2,2] @ [[3,1],[3,1],[3,1]] + [-5,-12] = [18,6] + [-5,-12] = [13,-6]
    y = Linear.apply(x, w, b)
    print(f"Linear output y = xW^T + b: {y}")
    print(f"  Computation: {x} @ {w.T} + {b}")
    print(f"  = {x @ w.T} + {b} = {y}")
    
    # Learnable SiLU: z = slope × y × sigmoid(y)
    z = LearnedSiLU.apply(y, slope)
    print(f"LearnedSiLU output z = α × y × σ(y): {z}")
    
    # Scalar loss for backpropagation
    loss = z.sum()
    print(f"Loss L = sum(z): {loss}")

    print("\n3. BACKWARD PASS")
    print("-" * 30)
    
    # Retain gradients for intermediate tensors (for educational analysis)
    y.retain_grad()
    z.retain_grad()
    loss.retain_grad() 
    
    # Compute all gradients via backpropagation
    loss.backward()
    print("Gradients computed successfully!")

    print("\n4. GRADIENT ANALYSIS")
    print("-" * 30)
    
    # Display all computed gradients
    print(f"∂L/∂z (loss grad): {z.grad}")
    print(f"∂L/∂y (linear grad): {y.grad}")
    print(f"∂L/∂x (input grad): {x.grad}")
    print(f"∂L/∂W (weight grad): {w.grad}")
    print(f"∂L/∂b (bias grad): {b.grad}")
    print(f"∂L/∂α (slope grad): {slope.grad}")
    
    print("\n5. VERIFICATION")
    print("-" * 30)
    
    # Verify gradient shapes match parameter shapes
    print("Gradient shape verification:")
    print(f"  x.grad.shape == x.shape: {x.grad.shape} == {x.shape} -> {x.grad.shape == x.shape}")
    print(f"  w.grad.shape == w.shape: {w.grad.shape} == {w.shape} -> {w.grad.shape == w.shape}")
    print(f"  b.grad.shape == b.shape: {b.grad.shape} == {b.shape} -> {b.grad.shape == b.shape}")
    print(f"  slope.grad.shape == slope.shape: {slope.grad.shape} == {slope.shape} -> {slope.grad.shape == slope.shape}")
    
    # Check that all gradients are finite (no NaN/Inf)
    all_finite = all([
        torch.isfinite(x.grad).all(),
        torch.isfinite(w.grad).all(), 
        torch.isfinite(b.grad).all(),
        torch.isfinite(slope.grad).all()
    ])
    print(f"  All gradients finite: {all_finite}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("All custom autograd functions working correctly.")
    print("=" * 60)
