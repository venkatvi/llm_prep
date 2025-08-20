# PyTorch Context Object (`ctx`) in Custom Autograd Functions

## Overview

In PyTorch's autograd system, `ctx` is a **context object** of type `torch.autograd.function.FunctionCtx` that serves as a communication bridge between the forward and backward passes of custom autograd functions.

## Type Definition

```python
ctx: torch.autograd.function.FunctionCtx
```

## Purpose and Role

The context object enables:
- **Information Transfer**: Pass data from forward pass to backward pass
- **Memory Management**: Efficiently store only necessary tensors for gradient computation
- **Temporal Bridge**: Connect forward computation with gradient calculation

## Core Methods

### `ctx.save_for_backward(*tensors)`

**Purpose**: Save tensors during forward pass for use in backward pass

**Usage**:
```python
@staticmethod
def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)  # Save multiple tensors
    return input @ weight
```

**Key Features**:
- Automatically handles memory management
- Preserves tensor metadata (requires_grad, device, etc.)
- Optimizes storage for gradient computation

### `ctx.saved_tensors`

**Purpose**: Retrieve tensors saved during forward pass

**Type**: `Tuple[torch.Tensor, ...]`

**Usage**:
```python
@staticmethod
def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors  # Unpack in save order
    grad_input = grad_output @ weight.T
    grad_weight = input.T @ grad_output
    return grad_input, grad_weight
```

## Complete Example: Power Function

```python
class Power(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, 
                input: torch.Tensor, 
                n: int) -> torch.Tensor:
        """
        Compute x^n and save input for backward pass.
        
        Args:
            ctx: Context object for saving state
            input: Input tensor x
            n: Power exponent (n >= 2)
            
        Returns:
            output: x^n
        """
        ctx.save_for_backward(input)  # Save input tensor
        ctx.n = n  # Save non-tensor constants
        assert n >= 2, "Power must be >= 2"
        output = input ** n
        return output
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, 
                 grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Compute gradient: d/dx(x^n) = n * x^(n-1).
        
        Args:
            ctx: Context object with saved state
            grad_output: Gradient from next layer
            
        Returns:
            grad_input: Gradient w.r.t. input
            None: No gradient for integer parameter n
        """
        (input,) = ctx.saved_tensors  # Retrieve saved input
        n = ctx.n  # Retrieve saved constant
        
        # Compute local gradient: d/dx(x^n) = n * x^(n-1)
        self_grad = n * input ** (n - 1)
        
        # Apply chain rule
        grad_input = grad_output * self_grad
        
        # Return gradients for each input (None for non-tensor inputs)
        return grad_input, None
```

## Advanced Context Usage

### Saving Non-Tensor Data

```python
@staticmethod
def forward(ctx, input, scale_factor):
    ctx.save_for_backward(input)
    ctx.scale_factor = scale_factor  # Save scalar
    ctx.input_shape = input.shape    # Save metadata
    return input * scale_factor
```

### Multiple Tensor Handling

```python
@staticmethod
def forward(ctx, x, y, z):
    ctx.save_for_backward(x, y, z)  # Save multiple tensors
    return x * y + z

@staticmethod
def backward(ctx, grad_output):
    x, y, z = ctx.saved_tensors  # Unpack in order
    grad_x = grad_output * y
    grad_y = grad_output * x  
    grad_z = grad_output
    return grad_x, grad_y, grad_z
```

## Memory Management Benefits

### Without Context (Inefficient)
```python
# BAD: Keeping entire tensors in memory
class BadSquare(torch.autograd.Function):
    saved_input = None  # Global variable - memory leak!
    
    @staticmethod
    def forward(ctx, input):
        BadSquare.saved_input = input  # Keeps tensor alive
        return input ** 2
```

### With Context (Efficient)
```python
# GOOD: Proper memory management
class GoodSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # Managed by PyTorch
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors  # Retrieved when needed
        return 2 * input * grad_output
```

## Type Annotations in Practice

### Forward Pass
```python
@staticmethod
def forward(ctx: torch.autograd.function.FunctionCtx, 
            input: torch.Tensor, 
            weight: torch.Tensor) -> torch.Tensor:
    ctx.save_for_backward(input, weight)
    return torch.matmul(input, weight)
```

### Backward Pass
```python
@staticmethod
def backward(ctx: torch.autograd.function.FunctionCtx, 
             grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input, weight = ctx.saved_tensors
    grad_input = torch.matmul(grad_output, weight.T)
    grad_weight = torch.matmul(input.T, grad_output)
    return grad_input, grad_weight
```

## Best Practices

### 1. Always Use Type Hints
```python
def forward(ctx: torch.autograd.function.FunctionCtx, 
            input: torch.Tensor) -> torch.Tensor:
```

### 2. Save Only What You Need
```python
# GOOD: Save only required tensors
ctx.save_for_backward(input)  # Only input needed for gradient

# BAD: Save unnecessary tensors
ctx.save_for_backward(input, intermediate, output)  # Wastes memory
```

### 3. Use Descriptive Unpacking
```python
# GOOD: Clear variable names
(input, weight, bias) = ctx.saved_tensors

# BAD: Confusing unpacking
(a, b, c) = ctx.saved_tensors
```

### 4. Handle None Gradients
```python
@staticmethod
def backward(ctx, grad_output):
    if grad_output is None:
        return None
    # ... compute gradients
```

## Common Patterns

### Single Input Function
```python
@staticmethod
def forward(ctx, input):
    ctx.save_for_backward(input)
    return some_operation(input)

@staticmethod  
def backward(ctx, grad_output):
    (input,) = ctx.saved_tensors  # Note the comma for single tuple
    return grad_output * derivative(input)
```

### Multi-Input Function
```python
@staticmethod
def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x * y

@staticmethod
def backward(ctx, grad_output):
    x, y = ctx.saved_tensors
    return grad_output * y, grad_output * x  # Gradients for x, y
```

## Debugging Context

### Print Saved Tensors
```python
@staticmethod
def backward(ctx, grad_output):
    print(f"Saved tensors: {len(ctx.saved_tensors)}")
    for i, tensor in enumerate(ctx.saved_tensors):
        print(f"Tensor {i}: shape={tensor.shape}, device={tensor.device}")
    # ... rest of backward pass
```

### Verify Context State
```python
@staticmethod
def forward(ctx, input):
    ctx.save_for_backward(input)
    ctx.debug_info = f"Forward called with input shape: {input.shape}"
    return input ** 2

@staticmethod
def backward(ctx, grad_output):
    print(ctx.debug_info)  # Access saved debug information
    (input,) = ctx.saved_tensors
    return 2 * input * grad_output
```

## Summary

The context object `ctx` is PyTorch's elegant solution for:

- **Bridging Time**: Connecting forward computation with backward gradient calculation
- **Managing Memory**: Efficiently storing only necessary tensors
- **Enabling Gradients**: Providing access to inputs needed for derivative computation
- **Type Safety**: Clear interfaces with proper type annotations

Understanding `ctx` is fundamental to creating custom autograd functions that integrate seamlessly with PyTorch's automatic differentiation system.