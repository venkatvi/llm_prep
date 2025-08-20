# Custom PyTorch Autograd Implementation

A comprehensive collection of custom PyTorch autograd functions demonstrating gradient computation for fundamental mathematical operations, neural network layers, and activation functions.

## Overview

This module provides educational implementations of core deep learning components using PyTorch's `autograd.Function` interface. Each implementation includes both forward computation and custom backward gradient calculation, demonstrating the mathematical foundations of automatic differentiation.

## Features

- **Mathematical Functions**: Basic operations (square, cube, exponential) with analytical gradients
- **Neural Network Layers**: Custom linear layer with matrix multiplication gradients  
- **Activation Functions**: Standard activations (ReLU, Sigmoid, Tanh) plus learnable variants
- **Educational Focus**: Clear documentation showing mathematical derivations
- **Gradient Verification**: Complete backward pass implementations with proper chain rule application

## File Structure

### Core Modules

- **`simple.py`** - Basic mathematical functions (Square, Cube, Exp)
- **`linear.py`** - Linear layer implementation with matrix multiplication  
- **`activations.py`** - Neural network activation functions
- **`main.py`** - Demonstration script combining multiple custom functions

### Dependencies

- PyTorch 2.0+
- Python 3.8+

## Mathematical Operations

### Simple Functions (`simple.py`)

#### Square Function
```python
# Forward: f(x) = x^2
# Backward: df/dx = 2x
result = Square.apply(x)
```

#### Cube Function  
```python
# Forward: f(x) = x^3
# Backward: df/dx = 3x^2
result = Cube.apply(x)
```

#### Exponential Function
```python
# Forward: f(x) = e^x
# Backward: df/dx = e^x
result = Exp.apply(x)
```

### Linear Layer (`linear.py`)

Custom implementation of fully connected layer:

```python
# Forward: y = xW^T + b
# Gradients:
#   ∂L/∂x = ∂L/∂y @ W
#   ∂L/∂W = ∂L/∂y^T @ x  
#   ∂L/∂b = sum(∂L/∂y, dim=0)

output = Linear.apply(input, weights, bias)
```

**Input Shapes**:
- `input`: [batch_size, input_features]
- `weights`: [output_features, input_features]  
- `bias`: [batch_size, output_features]

### Activation Functions (`activations.py`)

#### ReLU
```python
# Forward: f(x) = max(0, x)
# Backward: df/dx = 1 if x > 0, else 0
output = ReLU.apply(input)
```

#### Sigmoid
```python
# Forward: f(x) = 1 / (1 + e^(-x))
# Backward: df/dx = sigmoid(x) * (1 - sigmoid(x))
output = Sigmoid.apply(input)
```

#### Hyperbolic Tangent
```python
# Forward: f(x) = tanh(x)
# Backward: df/dx = 1 - tanh^2(x)
output = Tanh.apply(input)
```

#### Learnable SiLU
```python
# Forward: f(x) = slope * x * sigmoid(x)
# Gradients:
#   ∂f/∂x = slope * [sigmoid(x) + x * sigmoid'(x)]
#   ∂f/∂slope = x * sigmoid(x)
output = LearnedSiLU.apply(input, slope)
```

## Usage Examples

### Basic Mathematical Operations

```python
import torch
from simple import Square, Cube, Exp

# Create input with gradient tracking
x = torch.tensor(2.0, requires_grad=True)

# Forward pass
y_square = Square.apply(x)      # 4.0
y_cube = Cube.apply(x)          # 8.0  
y_exp = Exp.apply(x)            # e^2 ≈ 7.39

# Backward pass
y_square.backward()
print(x.grad)  # 4.0 (derivative of x^2 at x=2)
```

### Neural Network Layer

```python
import torch
from linear import Linear
from activations import ReLU

# Create input and parameters
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # [1, 3]
W = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], requires_grad=True)  # [2, 3]
b = torch.tensor([[0.1, 0.2]], requires_grad=True)  # [1, 2]

# Forward pass
linear_out = Linear.apply(x, W, b)  # [1, 2]
activation_out = ReLU.apply(linear_out)

# Backward pass
loss = activation_out.sum()
loss.backward()

print(f"Input gradient: {x.grad}")
print(f"Weight gradient: {W.grad}")
print(f"Bias gradient: {b.grad}")
```

### Learnable Activation Function

```python
import torch
from activations import LearnedSiLU

# Input and learnable parameter
x = torch.tensor([[1.0, -1.0, 2.0]], requires_grad=True)
slope = torch.tensor([0.5], requires_grad=True)

# Forward pass with learnable activation
y = LearnedSiLU.apply(x, slope)

# Backward pass
loss = y.sum()
loss.backward()

print(f"Input gradient: {x.grad}")
print(f"Slope gradient: {slope.grad}")  # Shows how to update learnable parameter
```

### Complete Neural Network Example

```python
import torch
from linear import Linear
from activations import LearnedSiLU

# Network parameters
x = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)
W = torch.tensor([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]], requires_grad=True)
b = torch.tensor([[-5.0, -12.0]], requires_grad=True)
slope = torch.tensor([-1.0], requires_grad=True)

# Forward pass
linear_output = Linear.apply(x, W, b)           # Linear transformation
final_output = LearnedSiLU.apply(linear_output, slope)  # Learnable activation

# Loss and backward pass
loss = final_output.sum()
loss.backward()

# All gradients computed automatically
print("Gradients computed for all parameters!")
```

## Mathematical Foundations

### Chain Rule Implementation

Each custom function implements the chain rule for backpropagation:

```
∂L/∂input = ∂L/∂output * ∂output/∂input
```

Where:
- `∂L/∂output` is `grad_output` (received from next layer)
- `∂output/∂input` is the local gradient (computed in backward method)

### Matrix Calculus for Linear Layer

The linear layer demonstrates matrix calculus for neural networks:

**Forward**: `y = xW^T + b`

**Gradients**:
- `∂L/∂x = ∂L/∂y @ W` (gradient flows back to input)
- `∂L/∂W = ∂L/∂y^T @ x` (weight updates)  
- `∂L/∂b = sum(∂L/∂y, dim=0)` (bias updates)

### Product Rule for Complex Functions

The LearnedSiLU demonstrates the product rule for `f(x) = slope * x * sigmoid(x)`:

```
∂f/∂x = slope * [sigmoid(x) + x * sigmoid'(x)]
```

## Educational Benefits

### Understanding Automatic Differentiation
- See how PyTorch computes gradients under the hood
- Learn the mathematical derivations behind common operations
- Understand the relationship between forward and backward passes

### Debugging Neural Networks
- Custom gradients help identify gradient flow issues
- Clear implementations make debugging easier
- Educational comments explain each step

### Extending PyTorch
- Learn how to implement custom operations
- Understand the `autograd.Function` interface
- Foundation for research and novel architectures

## Performance Notes

These implementations are for **educational purposes** and may not be as optimized as PyTorch's built-in functions. For production use, prefer:

- `torch.nn.Linear` instead of custom Linear
- `torch.nn.ReLU` instead of custom ReLU  
- `torch.nn.functional` for standard operations

## Testing Gradient Correctness

PyTorch provides tools to verify custom gradients:

```python
import torch
from torch.autograd import gradcheck

# Test custom function gradients
def test_square_gradients():
    x = torch.randn(10, dtype=torch.double, requires_grad=True)
    test = gradcheck(Square.apply, x, eps=1e-6, atol=1e-4)
    print(f"Square gradients correct: {test}")

test_square_gradients()
```

## Future Extensions

Potential additions to this educational framework:

- **Convolutional operations** with spatial gradients
- **Attention mechanisms** with custom backward passes
- **Normalization layers** (BatchNorm, LayerNorm)
- **Advanced optimizers** with custom parameter updates
- **Memory-efficient operations** with checkpointing

## References

- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [Matrix Calculus for Deep Learning](http://explained.ai/matrix-calculus/)
- [Automatic Differentiation in Machine Learning: A Survey](https://arxiv.org/abs/1502.05767)

## License

Copyright (c) 2025. All rights reserved.