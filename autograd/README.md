# Custom PyTorch Autograd Implementation

Educational implementations of PyTorch autograd functions with custom forward/backward passes.

## Modules

- **simple.py**: Mathematical functions (Power, Square, Cube, Exp)
- **linear.py**: Linear layer implementation  
- **activations.py**: Activation functions (Tanh, Sigmoid, ReLU, LearnedSiLU)
- **main.py**: Integration demo

## Usage

```python
import torch
from simple import Square
from linear import Linear
from activations import ReLU

x = torch.tensor(3.0, requires_grad=True)
y = Square.apply(x)  # y = 9.0
y.backward()
print(x.grad)  # 6.0
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