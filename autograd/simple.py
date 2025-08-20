"""
Mathematical functions with custom autograd implementation.

Classes: Power, Square, Cube, Exp
"""

from typing import Tuple
import torch 

class Power(torch.autograd.Function):
    """Compute x^n with gradients for both inputs."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input, n)
        return input ** n
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input, n = ctx.saved_tensors
        grad_input = grad_output * n * input ** (n - 1)
        grad_n = grad_output * input ** n * torch.log(input)
        return grad_input, grad_n
        

class Square(torch.autograd.Function):
    """Compute x^2 with gradient 2x."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        return grad_output * 2 * input

class Cube(torch.autograd.Function): 
    """Compute x^3 with gradient 3x^2."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input ** 3
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor: 
        (input, ) = ctx.saved_tensors
        return grad_output * 3 * input ** 2
    
class Exp(torch.autograd.Function): 
    """Compute e^x with gradient e^x."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor: 
        output = torch.exp(input)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (output, ) = ctx.saved_tensors
        return grad_output * output