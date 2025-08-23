# Custom PyTorch Autograd Implementation

Educational implementations of PyTorch autograd functions with custom forward/backward passes, comprehensive type annotations, and enhanced documentation.

## Modules

- **simple.py**: Mathematical functions with type annotations (Power, Square, Cube, Exp)
- **linear.py**: Linear layer implementation with comprehensive type hints
- **activations.py**: Activation functions with type safety (Tanh, Sigmoid, ReLU, LearnedSiLU)
- **main.py**: Integration demo showcasing type-safe autograd operations

## Usage

```python
import torch
from typing import Optional
from simple import Square
from linear import Linear
from activations import ReLU

# Type-safe tensor operations with gradient tracking
x: torch.Tensor = torch.tensor(3.0, requires_grad=True)
y: torch.Tensor = Square.apply(x)  # y = 9.0
y.backward()
print(x.grad)  # 6.0 (type: Optional[torch.Tensor])

# Complex autograd chain with type annotations
input_tensor: torch.Tensor = torch.randn(2, 3, requires_grad=True)
weights: torch.Tensor = torch.randn(4, 3, requires_grad=True)
bias: torch.Tensor = torch.randn(2, 4, requires_grad=True)

# Forward pass with custom autograd functions
linear_output: torch.Tensor = Linear.apply(input_tensor, weights, bias)
activated: torch.Tensor = ReLU.apply(linear_output)
squared: torch.Tensor = Square.apply(activated)
loss: torch.Tensor = squared.sum()

# Backward pass computes gradients automatically
loss.backward()
```

## Testing

```bash
cd tests && python run_tests.py  # All 72 tests
pytest tests/ -v                 # With pytest
```

## Key Functions

| Function | Forward | Gradient |
|----------|---------|----------|
| Power | x^n | n·x^(n-1), x^n·ln(x) |
| Square | x² | 2x |
| Tanh | tanh(x) | 1 - tanh²(x) |
| ReLU | max(0,x) | 1 if x>0 else 0 |
| Linear | xW^T + b | Various matrix gradients |

## Enhanced Code Quality

### Type Annotations
All autograd implementations include comprehensive type hints:
- **Function Signatures**: Typed `forward()` and `backward()` methods
- **Context Objects**: Proper typing for gradient context management
- **Tensor Operations**: Explicit tensor type annotations throughout
- **Optional Handling**: Proper typing for optional gradients and parameters

### Documentation Improvements
- **Docstrings**: Complete documentation for all custom autograd functions
- **Mathematical Formulas**: Clear documentation of forward and backward computations
- **Usage Examples**: Type-annotated examples showing proper usage patterns
- **Error Handling**: Enhanced error messages with type information

### Code Organization
- **Import Structure**: PEP8-compliant import organization
- **Variable Naming**: Clear, type-hinted variable names
- **Function Structure**: Consistent typing patterns across all modules
- **Testing Integration**: Type-safe test implementations with 72+ comprehensive tests