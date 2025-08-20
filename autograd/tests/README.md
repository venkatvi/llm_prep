# Autograd Tests

Comprehensive test suite for custom PyTorch autograd implementations.

## Test Structure

### `test_simple.py`
Tests for basic mathematical functions:
- **TestPower**: Power function f(x,n) = x^n
- **TestSquare**: Square function f(x) = x^2  
- **TestCube**: Cube function f(x) = x^3
- **TestExp**: Exponential function f(x) = e^x
- **TestTensorOperations**: Multi-element tensor operations

### `test_linear.py`
Tests for linear layer implementation:
- **TestLinear**: Linear transformation f(x,W,b) = xW^T + b
- **TestLinearEdgeCases**: Edge cases and various tensor dimensions

### `test_activations.py`
Tests for activation functions:
- **TestTanh**: Hyperbolic tangent activation
- **TestSigmoid**: Sigmoid activation  
- **TestReLU**: Rectified Linear Unit activation
- **TestLearnedSiLU**: Learnable SiLU activation
- **TestActivationConsistency**: Consistency with PyTorch implementations
- **TestActivationEdgeCases**: Edge cases and numerical stability

### `test_main.py`
Integration tests:
- **TestIntegration**: Multi-function pipeline testing
- **TestMainScript**: Main script execution validation

## Running Tests

### Run All Tests
```bash
cd autograd/tests
python run_tests.py
```

### Run Specific Categories
```bash
# Mathematical functions only
python run_tests.py --category simple

# Linear layer only  
python run_tests.py --category linear

# Activation functions only
python run_tests.py --category activations

# Integration tests only
python run_tests.py --category integration
```

### Verbose Output
```bash
python run_tests.py --verbose
```

### Individual Test Files
```bash
# Run specific test file
python test_simple.py
python test_linear.py
python test_activations.py  
python test_main.py
```

## Test Coverage

Each test file includes:
- **Forward pass verification**: Correct mathematical computation
- **Gradient correctness**: Analytical gradient validation
- **Gradient checking**: PyTorch's `gradcheck` for numerical verification
- **Edge cases**: Zero inputs, large values, negative inputs
- **Shape consistency**: Proper tensor dimension handling
- **Error handling**: Invalid input detection

## Test Features

- **Mathematical Accuracy**: Verifies correct implementation of derivatives
- **Numerical Stability**: Tests with various input ranges and edge cases
- **Shape Validation**: Ensures proper tensor broadcasting and dimensions
- **Gradient Flow**: Validates end-to-end gradient computation
- **PyTorch Consistency**: Compares with built-in PyTorch functions where applicable
- **Error Detection**: Tests proper error handling for invalid inputs